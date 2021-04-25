#include "dataloader.h"

using namespace std;

DataLoader::DataLoader(string dataSetPath, bool trainFlag, const string& kind, const string& revpct) {
    
    m_dataSetPath = dataSetPath;
    m_trainFlag = trainFlag;
    m_kind = kind;
    m_revpct = revpct;

    if (m_kind=="features") {
	    string dataPointName;
	    string dataDimName;
	    string dataValueName;

	    if (m_trainFlag) {
		    dataPointName = m_dataSetPath + "_trDataPoint.csv";
		    dataDimName   = m_dataSetPath + "_trDataDim.csv";
		    dataValueName = m_dataSetPath + "_trDataValue.csv";
	    } else {
		    dataPointName = m_dataSetPath + "_teDataPoint.csv";
		    dataDimName   = m_dataSetPath + "_teDataDim.csv";
		    dataValueName = m_dataSetPath + "_teDataValue.csv";
	    }

	    vector<int> dataPoint = loadFileInt(dataPointName);
	    vector<int> dataDim = loadFileInt(dataDimName);
	    vector<float> dataValue = loadFileFloat(dataValueName);
        cerr<<"Feature Value Counts:"<<endl;
        cerr<<dataPoint.size()<<endl;
        cerr<<dataDim.size()<<endl;
        cerr<<dataValue.size()<<endl;
        prepareFeatureData(dataPoint, dataDim, dataValue);
        cerr<<"Feature nRows"<<endl;
        cerr<<m_dataPoints.size()<<endl;
	
    } else if (m_kind=="labels") { // label
        string labelDimName;
        string labelPointName;

        if (m_trainFlag == true) { // train
            labelPointName = m_dataSetPath + "_trLabelPoint.csv";
            labelDimName = m_dataSetPath + "_trLabelDim.csv";
        } else { // test
            labelPointName = m_dataSetPath + "_teLabelPoint.csv";
            labelDimName = m_dataSetPath + "_teLabelDim.csv";
        }

        vector<int> labelPoint = loadFileInt(labelPointName);
        vector<int> labelDim = loadFileInt(labelDimName);
        cerr<<"Label Value Counts:"<<endl;
        cerr<<labelPoint.size()<<endl;
        cerr<<labelDim.size()<<endl;
        prepareLabelData(labelPoint, labelDim); //Creates m_datapoints vector
        cerr<<"Label nRows"<<endl;
        cerr<<m_dataPoints.size()<<endl;
    }

    else if (m_kind=="revealed_labels") {
        string labelDimNameRev;
        string labelPointNameRev;
        if (m_trainFlag==true) {
            labelPointNameRev = m_dataSetPath + "_trLabelPoint" + m_revpct + ".csv";
            labelDimNameRev = m_dataSetPath + "_trLabelDim" + m_revpct + ".csv";
        }
        else {
            labelPointNameRev = m_dataSetPath + "_teLabelPoint" + m_revpct + ".csv";
            labelDimNameRev = m_dataSetPath + "_teLabelDim" + m_revpct + ".csv";
        }

        vector<int> labelPointRev = loadFileInt(labelPointNameRev);
        vector<int> labelDimRev = loadFileInt(labelDimNameRev);
        cerr<<"Revealed Label Value Counts:"<<endl;
        cerr<<labelPointRev.size()<<endl;
        cerr<<labelDimRev.size()<<endl;
        prepareLabelData(labelPointRev, labelDimRev); 
    }
    else if (m_kind=="label_embeddings") {
        string lfPointName;
	    string lfDimName;
	    string lfValueName;
        lfPointName = m_dataSetPath + "_LFPoint.csv";
        lfDimName   = m_dataSetPath + "_LFDim.csv";
        lfValueName = m_dataSetPath + "_LFValue.csv";


	    vector<int> lfPoint = loadFileInt(lfPointName);
	    vector<int> lfDim = loadFileInt(lfDimName);
	    vector<float> lfValue = loadFileFloat(lfValueName);
        cerr<<"Loaded Label Feature File\n";
        cerr<<"LF Value Counts:"<<endl;
        cerr<<lfPoint.size()<<endl;
        cerr<<lfDim.size()<<endl;
        cerr<<lfValue.size()<<endl;
        prepareFeatureData(lfPoint, lfDim, lfValue);
        cerr<<"Created Label Feature Data\n";
    }
}

vector<float> DataLoader::loadFileFloat(string filePath) {
    ifstream dataPointFile(filePath);
    vector<float> out;
    for (float x; dataPointFile >> x;) {
        out.push_back(x);
    }
    return out;
}

vector<int> DataLoader::loadFileInt(string filePath) {
    ifstream dataPointFile(filePath);
    vector<int> out;
    for (int x; dataPointFile >> x;) {
        out.push_back(x);
    }
    return out;
}

void DataLoader::prepareFeatureData(vector<int> dataPoint, vector<int> dataDim, vector<float> dataValue) {
    int nbOfPoint = dataPoint[dataPoint.size() - 1];
    m_dim = *max_element(dataDim.begin(), dataDim.end()) + 1;
    cerr<<m_kind<<" num dims: "<<m_dim<<endl;
    
    size_t iter = 0;
    for (int i = 1; i <= nbOfPoint; i++)  {
        vector<int> tmpIndeces; 
        vector<float> tmpValues; 

        bool missingLabel = true;
        while (iter < dataPoint.size() && dataPoint[iter] == i) {
            missingLabel = false;
            tmpIndeces.push_back(dataDim[iter] - 1);
            tmpValues.push_back(dataValue[iter]);
            iter++;
        }

        if (!missingLabel) {
            tmpIndeces.push_back(m_dim - 1);
            tmpValues.push_back(1);
            DataPoint rowData(tmpIndeces, tmpValues);
            m_dataPoints.push_back(rowData);
        }
    }
    cerr<<m_kind<<" num rows loaded: "<<m_dataPoints.size()<<endl;
}

void DataLoader::prepareLabelData(vector<int> labelPoint, vector<int> labelDim) {
    m_nbOfClasses = *max_element(labelDim.begin(), labelDim.end());

    int nbOfPoint = labelPoint[labelPoint.size() - 1];

    cerr<<m_kind<<" num classes: "<<m_nbOfClasses<<endl;

    size_t iter = 0;
    for (int i = 1; i <= nbOfPoint; i++)  {
        vector<int> tmp;

        bool missingLabel = true;
        while (iter < labelPoint.size() && labelPoint[iter] == i) {
            missingLabel = false;
            tmp.push_back(labelDim[iter] - 1);
            iter++;
        }

        if (!missingLabel) {
            DataPoint rowData(tmp);
            m_dataPoints.push_back(rowData);
        }
    } 
    cerr<<m_kind<<" num rows loaded: "<<m_dataPoints.size()<<endl;
}