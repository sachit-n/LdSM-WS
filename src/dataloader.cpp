#include "dataloader.h"

using namespace std;

DataLoader::DataLoader(string dataSetPath, const string& kind, const bool trainFlag, const string& revpct) {
    
    m_dataSetPath = dataSetPath;
    m_kind = kind;
    m_trainFlag = trainFlag;
    m_revpct = revpct;

    string s = (m_trainFlag) ? "tr_" : "te_";

    if (m_kind=="features") {
	    string dataPointName;
	    string dataDimName;
	    string dataValueName;

        dataPointName = m_dataSetPath + s + "x_ix.csv";
        dataDimName   = m_dataSetPath + s + "x_col_ix.csv";
        dataValueName = m_dataSetPath + s + "x_v.csv";

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
	
    } else if (m_kind=="tr_labels") { // label
        string labelDimName;
        string labelPointName;

        labelPointName = m_dataSetPath + "tr_y_ix_" + m_revpct + ".csv";
        labelDimName = m_dataSetPath + "tr_y_col_ix_" + m_revpct + ".csv";
        cerr<<labelPointName<<endl;
        cerr<<labelDimName<<endl;

        vector<int> labelPoint = loadFileInt(labelPointName);
        vector<int> labelDim = loadFileInt(labelDimName);
        cerr<<"Label Value Counts:"<<endl;
        cerr<<labelPoint.size()<<endl;
        cerr<<labelDim.size()<<endl;
        prepareLabelData(labelPoint, labelDim); //Creates m_datapoints vector
    }

    else if (m_kind=="revealed_labels") {
        string labelDimName;
        string labelPointName;
        labelPointName = m_dataSetPath + "inc_" + m_revpct + "_te_y_ix.csv";
        labelDimName = m_dataSetPath + "inc_" + m_revpct + "_te_y_col_ix.csv";

        vector<int> labelPointRev = loadFileInt(labelPointName);
        vector<int> labelDimRev = loadFileInt(labelDimName);
        cerr<<"Revealed Label Value Counts:"<<endl;
        cerr<<labelPointRev.size()<<endl;
        cerr<<labelDimRev.size()<<endl;
        prepareLabelData(labelPointRev, labelDimRev); 
    }

    else if (m_kind=="hidden_labels") {
        string labelDimName;
        string labelPointName;
        labelPointName = m_dataSetPath + "exc_" + m_revpct + "_te_y_ix" ".csv";
        labelDimName = m_dataSetPath + "exc_" + m_revpct + "_te_y_col_ix" ".csv";

        vector<int> labelPointHidden = loadFileInt(labelPointName);
        vector<int> labelDimHidden = loadFileInt(labelDimName);
        cerr<<labelPointName<<endl;
        cerr<<"Hidden Label Value Counts:"<<endl;
        cerr<<labelPointHidden.size()<<endl;
        cerr<<labelDimHidden.size()<<endl;
        prepareLabelData(labelPointHidden, labelDimHidden);
    }


    else if (m_kind=="label_embeddings") {
        string lfPointName;
	    string lfDimName;
	    string lfValueName;
        lfPointName = m_dataSetPath + "w2v_emb_ix.csv";
        lfDimName   = m_dataSetPath + "w2v_emb_col_ix.csv";
        lfValueName = m_dataSetPath + "w2v_emb_v.csv";


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

    size_t iter = 0; //sachit: https://en.cppreference.com/w/cpp/types/size_t
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
        else {
            tmp = {};
            DataPoint rowData(tmp);
            m_dataPoints.push_back(rowData);
        }
    } 
    cerr<<m_kind<<" num rows loaded: "<<m_dataPoints.size()<<endl;
}