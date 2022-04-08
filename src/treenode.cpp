#include "treenode.h"
#include <float.h>

using namespace std;

vector<vector<float>> TreeNode::m_weight1;
vector<vector<float>> TreeNode::m_weight2;
vector<int> TreeNode::m_lastUpdate;
vector<int> TreeNode::m_dataClassCounter;
vector<float> TreeNode::m_probAllVector;
vector<vector<float>> TreeNode::m_probSingleVector;
vector<vector<float>> TreeNode::m_sNol;
vector<vector<float>> TreeNode::m_gNol;
vector<float> TreeNode::m_nNol;
vector<vector<int>> TreeNode::m_children_labelHistogram;

void TreeNode::initialize() {
	m_weight1.clear();
	m_weight2.clear();
	m_lastUpdate.clear();
	m_dataClassCounter.clear();
	m_probAllVector.clear();
	m_probSingleVector.clear();
	m_sNol.clear();
	m_gNol.clear();
	m_nNol.clear();
	m_children_labelHistogram.clear();

	m_weight1.resize(m_params->m, vector<float>(m_params->d, 0.f));
	m_weight2.resize(m_params->m, vector<float>(m_params->embSize, 0.f));
	m_lastUpdate.resize(m_params->d, 0);
	m_dataClassCounter.resize(m_params->k, 0); // number of data points in each class at each node
	m_probAllVector.resize(m_params->m, 0.f);
	m_probSingleVector.resize(m_params->m, vector<float>(m_params->k, 0.f));
	m_sNol.resize(m_params->m, vector<float>(m_params->d, 0.f));
	m_gNol.resize(m_params->m, vector<float>(m_params->d, 0.f));
	m_nNol.resize(m_params->m, 0.f);
	m_children_labelHistogram.resize(m_params->m, vector<int>(m_params->k, 0));

}

//removed argument trLabel from this function as it is not used
void TreeNode::meanStdCalc(const DataLoader &trData) { // only for non-sparse data

	int dim = trData.getDim();
	int dataIndexSize = m_dataIndex.size();
	m_mean.resize(dim, 0.f);
	m_std.resize(dim, 0.f);

	for (int index = 0; index < dataIndexSize; index++) {
		const DataPoint& data = trData.getDataPoint(m_dataIndex[index]);
		const vector<int>& dataPointIndeces = data.getDataIndeces();
		const vector<float>& dataPointValues = data.getDataValues();
		for (size_t i = 0; i < dataPointIndeces.size(); i++) {
			int idx = dataPointIndeces[i];
			float val = dataPointValues[i];
			m_mean[idx] += val;
		}
	}

	for (int i = 0; i < dim; i++) {
		m_mean[i] /= dataIndexSize;
	}

	for (int index = 0; index < dataIndexSize; index++) {
		const DataPoint& data = trData.getDataPoint(m_dataIndex[index]);
		const vector<int>& dataPointIndeces = data.getDataIndeces();
		const vector<float>& dataPointValues = data.getDataValues();
		for (size_t i = 0; i < dataPointIndeces.size(); i++) {
			int idx = dataPointIndeces[i];
			float val = dataPointValues[i];
			m_std[idx] += (val - m_mean[idx])*(val - m_mean[idx]);
		}
	}

	for (int i = 0; i < dim; i++) {
		m_std[i] = sqrt(m_std[i] / (dataIndexSize - 1));
	}
	m_std[dim - 1] = 1.f;

}

float gradAbsoluteLoss(int label, float prediction) {
	float expNegPred = exp(-prediction);
	float gradient = 0.5f * expNegPred / (1.f + expNegPred) / (1.f + expNegPred);
	if (label == 1)
		return -gradient;

	return gradient;
}

float gradEntropyLoss(int label, float prediction) {
	return -(label - 1 / (1 + exp(-prediction)));
}

void decayReg(float &weight, float coefL1, float coefL2, int stepsCount) {
	float decayL1 = coefL1 * stepsCount;
	if (decayL1 > fabs(weight))
		weight = 0.f;
	else if (decayL1 < fabs(weight)) {
		if (weight > 0)
			weight -= decayL1;
		else
			weight += decayL1;
	}
	float decayL2 = 1.f;
	if (coefL2 != 0)
		decayL2 = pow(1.f - coefL2, stepsCount);
	weight *= decayL2;
}

void TreeNode::weightUpdate(const DataLoader &trData, const DataLoader &trLabel, const DataLoader &trRevLabel, const DataLoader &labelFeatures,
	vector<int>& rooLabelHist, int maxLabelRoot) {

	std::function<float(int, float)> calcGradient;
	if (m_params->entropyLoss)
		calcGradient = gradEntropyLoss;
	else
		calcGradient = gradAbsoluteLoss;

	int d = m_params->d;
	int embSize = m_params->embSize;
	float clip = 1; // for clipping data.dot(weight)

	if (m_params->coefL1 != 0 || m_params->coefL2 != 0)
		fill(m_lastUpdate.begin(), m_lastUpdate.end(), 0);
	fill(m_dataClassCounter.begin(), m_dataClassCounter.end(), 0);
	fill(m_probAllVector.begin(), m_probAllVector.end(), 0.f);
	fill(m_nNol.begin(), m_nNol.end(), 0.f);
	for (int m = 0; m < m_params->m; m++) {
		fill(m_sNol[m].begin(), m_sNol[m].end(), 0.f);
		fill(m_gNol[m].begin(), m_gNol[m].end(), 0.f);
		fill(m_probSingleVector[m].begin(), m_probSingleVector[m].end(), 0.f);
	}

	const float weightMag = 1.f / m_params->d;
	const float weightMag2 = 1.f / m_params->embSize;
	for (int m = 0; m < m_params->m; m++) {
		for (int j = 0; j < d; j++)
			m_weight1[m][j] = ((float)(rand() - RAND_MAX / 2) / RAND_MAX) * weightMag;
		for (int j = 0; j < embSize; j++)
			m_weight2[m][j] = ((float)(rand() - RAND_MAX / 2) / RAND_MAX) * weightMag2;
	}

	int dataCounter = 0;

	float J; // objective function initialization
	vector<int> yhat; // regressors' label
	int tStep = 0;
	for (int e = 0; e < m_params->epochs; e++) {
		for (size_t index = 0; index < m_dataIndex.size(); index++) {
			tStep++;

			float lrWeight = 0.0;
			const vector<int>& labelVector = trLabel.getDataPoint(m_dataIndex[index]).getLabelVector();
			int labelSize = labelVector.size();
			for (int l = 0; l < labelSize; l++) {
				int k = labelVector[l];
				lrWeight += (float)maxLabelRoot / (float)rooLabelHist[k];
				m_dataClassCounter[k]++;
			}
			dataCounter += labelSize;
			float alpha = 0.0;
			if (m_params->exampleLearn)
				alpha = m_params->alpha * lrWeight;
			else
				alpha = m_params->alpha;

			// optimum parameters
			float optJ = FLT_MAX;
			float optJPurity = FLT_MAX;
			vector<int> optYhat(m_params->m);

			for (int s = 1; s < (1 << m_params->m); s++) {
				vector<float> tmpProbAllVector(m_params->m, 0);
				vector<vector<float>> tmpProbSingleVector(m_params->m, vector<float>(labelSize, 0.f));
				vector<int> tmpYhat(m_params->m);

				for (int m = 0; m < m_params->m; m++) {
					int a = (1 << m);
					tmpYhat[m] = ((s & a) == 0) ? 0 : 1;
					tmpProbAllVector[m] = (m_probAllVector[m] * (dataCounter - labelSize) +
						labelSize * tmpYhat[m]) /
						(float)dataCounter;

					for (int l = 0; l < labelSize; l++) {
						int k = labelVector[l];
						tmpProbSingleVector[m][l] = (m_probSingleVector[m][k] *
							(m_dataClassCounter[k] - 1) + tmpYhat[m]) /
							(float)m_dataClassCounter[k];
					}
				}

				float tmpBalance = 0.0;
				for (int m1 = 0; m1 < m_params->m; m1++) {
					for (int m2 = m1 + 1; m2 < m_params->m; m2++) {
						tmpBalance += abs(tmpProbAllVector[m1] - tmpProbAllVector[m2]);
					}
				}

				float tmpBoth = 0.0;
				for (int m = 0; m < m_params->m; m++)
					tmpBoth += tmpProbAllVector[m];
				tmpBoth -= 1.0;
				if (tmpBoth < 0.0)
					tmpBoth = -tmpBoth;

				float tmpJPurity = 0.0;
				for (int l = 0; l < labelSize; l++) {
					int k = labelVector[l];
					float purity = 0.0;
					for (int m1 = 0; m1 < m_params->m; m1++) {
						for (int m2 = m1 + 1; m2 < m_params->m; m2++) {
							float tmp = tmpProbSingleVector[m1][l] - tmpProbSingleVector[m2][l];
							if (tmp < 0.0)
								tmp = -tmp;
							purity += tmp;
						}
					}
					tmpJPurity += purity * ((float)m_dataClassCounter[k] / (float)dataCounter);
				}
				float tmpJ = tmpBalance + m_params->l1 * tmpBoth - m_params->l2 * tmpJPurity;
				if (tmpJ == optJ && (s != (1 << m_params->m) - 1)) {
					int r = rand() % 2;
					if (r >= 1) {
						optJ = tmpJ;
						optJPurity = tmpJPurity;
						optYhat = tmpYhat;
					}
				}

				if (tmpJ < optJ) {
					optJ = tmpJ;
					optJPurity = tmpJPurity;
					optYhat = tmpYhat;
				}
			}

			J = optJ;
			yhat = optYhat;

			vector<float> newDotProduct(m_params->m);
			vector<float> newDotProduct1(m_params->m);
			vector<float> newDotProduct2(m_params->m);

			DataPoint dataNormal;
			if (m_params->sparse) {
				const DataPoint& dataNormal = trData.getDataPoint(m_dataIndex[index]);
				const vector<int>& dataPointIndeces = dataNormal.getDataIndeces();
				const vector<float>& dataPointValues = dataNormal.getDataValues();

				for (int m = 0; m < m_params->m; m++) {
					for (size_t i = 0; i < dataPointIndeces.size(); i++) {
						int idx = dataPointIndeces[i];
						float val = dataPointValues[i];
						if (m_params->coefL1 != 0 || m_params->coefL2 != 0) {
							decayReg(m_weight1[m][idx], m_params->coefL1, m_params->coefL2, tStep - m_lastUpdate[idx]);
						}
						if (abs(val) > m_sNol[m][idx]) {
							m_weight1[m][idx] = m_weight1[m][idx] * m_sNol[m][idx] / abs(val);
							m_sNol[m][idx] = abs(val);
						}
					}

					float dotProduct = dataNormal.dot(m_weight1[m]);

					for (size_t i = 0; i < dataPointIndeces.size(); i++) {
						int idx = dataPointIndeces[i];
						float val = dataPointValues[i];
						m_nNol[m] += (val * val) / (m_sNol[m][idx] * m_sNol[m][idx]);
					}

					float gradient_const = calcGradient(yhat[m], dotProduct);
					for (size_t i = 0; i < dataPointIndeces.size(); i++) {
						int idx = dataPointIndeces[i];
						float val = dataPointValues[i];
						float gradient = gradient_const * val;
						m_gNol[m][idx] += (gradient * gradient);
						if (m_gNol[m][idx] != 0)
							m_weight1[m][idx] -= alpha * sqrt(tStep / m_nNol[m]) * gradient /
							(m_sNol[m][idx] * sqrt(m_gNol[m][idx]));
					}
				}
				if (m_params->coefL1 != 0 || m_params->coefL2 != 0) {
				for (size_t i = 0; i < dataPointIndeces.size(); i++)
					m_lastUpdate[i] = tStep;
				}

				for (int m = 0; m < m_params->m; m++) {
					newDotProduct1[m] = dataNormal.dot(m_weight1[m]);
				}
			}
			else {
				DataPoint data = trData.getDataPoint(m_dataIndex[index]);
				DataPoint dataNormal = data.normal(m_mean, m_std);
				const vector<int>& dataPointIndeces = dataNormal.getDataIndeces();
				const vector<float>& dataPointValues = dataNormal.getDataValues();

				for (int m = 0; m < m_params->m; m++) {

					float dotProduct1 = dataNormal.dot(m_weight1[m]);
					float gradient_const = calcGradient(yhat[m], dotProduct1);
					for (size_t i = 0; i < dataPointIndeces.size(); i++) {
						int idx = dataPointIndeces[i];
						float val = dataPointValues[i];
						float gradient = gradient_const * val;
						m_weight1[m][idx] -= alpha * gradient;
					}
				}

				for (int m = 0; m < m_params->m; m++) {
					newDotProduct1[m] = dataNormal.dot(m_weight1[m]);
				}
			}
			// Train regressor 2 using SGD based on optimal directions to send example
			DataPoint revLabels = trRevLabel.getDataPoint(m_dataIndex[index]);
			DataPoint data2 = combineEmbeddings(revLabels, labelFeatures);
			// DataPoint dataNormal2 = data2.normal(m_mean, m_std); // ToDo - Test this module
			const vector<int>& dataPointIndeces2 = data2.getDataIndeces();
			const vector<float>& dataPointValues2 = data2.getDataValues();
			for (int m = 0; m < m_params->m; m++) {

				float dotProduct2 = data2.dot(m_weight2[m]);
				float gradient_const2 = calcGradient(yhat[m], dotProduct2);
				for (size_t i = 0; i < dataPointIndeces2.size(); i++) {
					int idx = dataPointIndeces2[i];
					float val = dataPointValues2[i];
					float gradient2 = gradient_const2 * val;
					m_weight2[m][idx] -= alpha * gradient2;
				}
			}
			for (int m = 0; m < m_params->m; m++) {
					newDotProduct2[m] = data2.dot(m_weight2[m]);
				}
			// Take the sum of the two dot products
			for (int m = 0; m < m_params->m; m++) {
				newDotProduct[m] = m_params->c1*newDotProduct1[m] + m_params->c2*newDotProduct2[m];
			}

			for (int m = 0; m < m_params->m; m++) {
				newDotProduct[m] = min(max(newDotProduct[m], 0.0f), clip);
				m_probAllVector[m] = (m_probAllVector[m] * (dataCounter - labelSize) +
					labelSize * newDotProduct[m]) / (float)dataCounter;

				for (int l = 0; l < labelSize; l++) {
					int k = labelVector[l];
					m_probSingleVector[m][k] = (m_probSingleVector[m][k] *
						(m_dataClassCounter[k] - 1) +
						newDotProduct[m]) / (float)m_dataClassCounter[k];
				}
			}


		}
	}

	const float weightTreshold = weightMag;
	const float weightTreshold2 = weightMag2;

	for (int m = 0; m < m_params->m; m++) {
		for (int j = 0; j < d; j++) {
			if ((m_params->coefL1 != 0 || m_params->coefL2 != 0) && m_lastUpdate[j] > 0) {
				decayReg(m_weight1[m][j], m_params->coefL1, m_params->coefL2, tStep - m_lastUpdate[j]);
			}
			if (fabs(m_weight1[m][j]) < weightTreshold) {
				m_weight1[m][j] = 0.f;
			}
		}
		for (int j = 0; j < embSize; j++) {
			if (fabs(m_weight2[m][j]) < weightTreshold2) {
				m_weight2[m][j] = 0.f;
			}
		}
	}
}


void TreeNode::destroyChildren(vector<TreeNode>& nodes) {
    for (int m = 0; m < m_params->m; m++) {
        int ch = m_children[m];
        nodes[ch].m_dataIndex.clear();
    }
    m_children.clear();
}

int TreeNode::makeChildren(const DataLoader &trData, const DataLoader &trLabel, const DataLoader &trRevLabel, const DataLoader &labelFeatures, const int& N, vector<TreeNode>& nodes) {

    m_children.resize(m_params->m); // m_params->m children

	for (int m = 0; m < m_params->m; m++) {
        fill(m_children_labelHistogram[m].begin(), m_children_labelHistogram[m].end(), 0);
    }

    for (int m = 0; m < m_params->m; m++) {
        m_children[m] = N + m;
        int ch = m_children[m];
    }

    for (size_t index = 0; index < m_dataIndex.size(); index++) {
		vector<float> dotProduct(m_params->m);
		if (m_params->sparse) {
			DataPoint dataNormal = trData.getDataPoint(m_dataIndex[index]);
			DataPoint revLabels = trRevLabel.getDataPoint(m_dataIndex[index]);
			DataPoint data2 = combineEmbeddings(revLabels, labelFeatures);
			for (int m = 0; m < m_params->m; m++) {
				dotProduct[m] = m_params->c1*dataNormal.dot(m_weight1[m]) + m_params->c2*data2.dot(m_weight2[m]);
			}
		}
        else {
			DataPoint data = trData.getDataPoint(m_dataIndex[index]);
			DataPoint dataNormal = data.normal(m_mean, m_std);
			DataPoint revLabels = trRevLabel.getDataPoint(m_dataIndex[index]);
			DataPoint data2 = combineEmbeddings(revLabels, labelFeatures);
			for (int m = 0; m < m_params->m; m++) {
				dotProduct[m] = m_params->c1*dataNormal.dot(m_weight1[m]) + m_params->c2*data2.dot(m_weight2[m]);
			}
		}
        vector<int> childIndicator(m_params->m, 0);
        for (int m = 0; m < m_params->m; m++) {

			int ch = m_children[m];
            if (dotProduct[m] >= 0.5) {
                childIndicator[m] = 1;
                nodes[ch].m_dataIndex.push_back(m_dataIndex[index]);
                const vector<int>& labelVector = trLabel.getDataPoint(m_dataIndex[index]).getLabelVector();
                for (size_t l = 0; l < labelVector.size(); l++) {
                    int k = labelVector[l];
					m_children_labelHistogram[m][k]++; //ToDo - All or Only Hidden. May try both
                }
            }
        }
        int cr = 0;
        for (int m = 0; m < m_params->m; m++)
            cr += childIndicator[m] * (1 << m);

        if (cr == 0) { // none direction
            float maxDot = *max_element(dotProduct.begin(), dotProduct.end());
            for (int m = 0; m < m_params->m; m++) {
                int ch = m_children[m];
                if (dotProduct[m] == maxDot) {
                    nodes[ch].m_dataIndex.push_back(m_dataIndex[index]);
                    const vector<int>& labelVector = trLabel.getDataPoint(
                                                m_dataIndex[index]).getLabelVector();
                    for (size_t l = 0; l < labelVector.size(); l++) {
                        int k = labelVector[l];
						m_children_labelHistogram[m][k]++; //ToDo - Only hidden labels update maybe?
                    }
                }
            }
        }
    }

    bool indic = true;
    for (int m = 0; m < m_params->m; m++) {
        int ch = m_children[m];
        if (m_dataIndex.size() == nodes[ch].m_dataIndex.size() || nodes[ch].m_dataIndex.empty())
            indic = false;
    }

    m_dataIndex.clear();
    m_dataIndex.resize(1);
    m_dataIndex.shrink_to_fit();

    if (indic == false) {
        destroyChildren(nodes);
        return 0;
    } else {
        for (int m = 0; m < m_params->m; m++) {
            int ch = m_children[m];
            nodes[ch].m_nodeId = ch;
            nodes[ch].m_depth = m_depth + 1;
			nodes[ch].m_labelHistogramSparse.set(m_children_labelHistogram[m]);
			m_weightSparse1.push_back(Varray<float>(m_weight1[m]));
			m_weightSparse2.push_back(Varray<float>(m_weight2[m])); //added
        }
    }

    return m_params->m;
}

void TreeNode::testBatch(const DataLoader &teData, const DataLoader &teRevLabel, const DataLoader &labelFeatures, vector<TreeNode>& nodes,
                         vector<vector<int>>& leafs) {

    if (isLeaf()) {
        for (size_t index = 0; index < m_dataIndex.size(); index++) {
            leafs[m_dataIndex[index]].push_back(m_nodeId);
        }
    } else {
        for (int m = 0; m < m_params->m; m++) {
            fill(m_weight1[m].begin(), m_weight1[m].end(), 0);
			fill(m_weight2[m].begin(), m_weight2[m].end(), 0);
        }
        for (int m = 0; m < m_params->m; m++) {
            nodes[m_children[m]].m_dataIndex.clear();
            nodes[m_children[m]].m_dataIndex.reserve(m_dataIndex.size());
            for (size_t i = 0; i < m_weightSparse1[m].myMap.index.size(); i++) {
                m_weight1[m][m_weightSparse1[m].myMap.index[i]] = m_weightSparse1[m].myMap.value[i];
            }
			for (size_t j = 0; j < m_weightSparse2[m].myMap.index.size(); j++) {
				m_weight2[m][m_weightSparse2[m].myMap.index[j]] = m_weightSparse2[m].myMap.value[j];
			}
        }

        for (size_t index = 0; index < m_dataIndex.size(); index++) {
            bool sent = false;
            int mMaxIdx = 0;
            float mMax = -1.0;
            for (int m = 0; m < m_params->m; m++) {
				float dotProduct;
				if (m_params->sparse) {
					const DataPoint& data = teData.getDataPoint(m_dataIndex[index]);
					DataPoint revLabels = teRevLabel.getDataPoint(m_dataIndex[index]);
					DataPoint data2 = combineEmbeddings(revLabels, labelFeatures);
					float dotProduct1 = data.dot(m_weight1[m]);
					float dotProduct2 = data2.dot(m_weight2[m]);
					dotProduct = m_params->c1*dotProduct1 + m_params->c2*dotProduct2;
				}
				else {
					DataPoint data = teData.getDataPoint(m_dataIndex[index]);
					DataPoint dataNormal = data.normal(m_mean, m_std);
					DataPoint revLabels = teRevLabel.getDataPoint(m_dataIndex[index]);
					DataPoint data2 = combineEmbeddings(revLabels, labelFeatures);
					dotProduct = m_params->c1*(dataNormal.dot(m_weight1[m])) + m_params->c2*(data2.dot(m_weight2[m]));
				}

                if (dotProduct > mMax) {
                    mMaxIdx = m;
                    mMax = dotProduct;
                }

                if (dotProduct >= 0.5) {
                    nodes[m_children[m]].m_dataIndex.push_back(m_dataIndex[index]);
                    sent = true;
                }
            }

            if (!sent) {
                nodes[m_children[mMaxIdx]].m_dataIndex.push_back(m_dataIndex[index]);
            }
        }
    }

    m_dataIndex.clear();
    m_dataIndex.resize(1);
    m_dataIndex.shrink_to_fit();
}

void TreeNode::addHistogram(labelEst& labelHistogramSum, int leafCount, const vector<int> &revLabels) const {
	int revLabelIx = 0;
	int k = revLabels.size();
    for (size_t i = 0; i < m_NormalLabelHistogramSparse.size(); i++) {
        int idx = m_NormalLabelHistogramSparse[i].first;
        float val = m_NormalLabelHistogramSparse[i].second;
		if (find(revLabels.begin(), revLabels.end(), idx) != revLabels.end()) {
        	// val = val*0.1;
			continue;
		}
		labelHistogramSum.regular[idx] += (val / leafCount);
    }
}

void TreeNode::normalizeLabelHist() {
	int sumLabel = 0;
	size_t labelSize = m_labelHistogramSparse.myMap.value.size();
	for (size_t i = 0; i < labelSize; i++)
		sumLabel += m_labelHistogramSparse.myMap.value[i];

	for (size_t i = 0; i < m_labelHistogramSparse.myMap.index.size(); i++) {
		m_NormalLabelHistogramSparse.push_back(make_pair(m_labelHistogramSparse.myMap.index[i],
			(float)m_labelHistogramSparse.myMap.value[i] / sumLabel));
	}
}

//ToDo - Test this function
DataPoint TreeNode::combineEmbeddings(const DataPoint& labels, const DataLoader& labelFeatures) {

	const vector<int>& labelIndeces = labels.getLabelVector();
    int nLabels = labels.size();

	// Creating label indeces vector (vector from 0 - embSize)
    vector<int> lfIndeces(m_params->embSize);
	std::iota (std::begin(lfIndeces), std::end(lfIndeces), 0); //Create vector [0,1,2,...,emb_size-1]. ToDo - test this works as expected

	// Adding embedding vectors of the revealed labels
    vector<float> lfValues(m_params->embSize, 0.0);
    for (int l = 0; l < nLabels; l++) {
        int k = labelIndeces[l]; //class index
        const vector<float> labelFeature = labelFeatures.getDataPoint(k).getDataValues(); //embedding of class index . ToDo - Test this works as expected
        std::transform (lfValues.begin(), lfValues.end()-1, labelFeature.begin(), lfValues.begin(), std::plus<float>()); // adding embedding of label l to lfValues vector
    }

	// Calculating the norm of the combined label feature vector, and dividing lfValues by its norm. ToDo - Test it works as expected
	float lfNorm = sqrt(inner_product(lfValues.begin(), lfValues.end(), lfValues.begin(), 0.0L));

	float mltp = 1/lfNorm;
	std::transform(lfValues.begin(), lfValues.end(), lfValues.begin(),
               std::bind(std::multiplies<float>(), std::placeholders::_1, mltp));

	//Last feature has to be 1 for bias
	lfValues[m_params->embSize-1] = 1;

    DataPoint lfData(lfIndeces, lfValues);
    return lfData;
}