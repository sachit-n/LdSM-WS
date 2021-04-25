#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <set>
#include <random>

#include "datapoint.h"
#include "utils.h"

using namespace std;

class DataLoader {

private:

    string m_dataSetPath;
    string m_dataSetName; 
    bool m_trainFlag; 
    string m_kind;
    string m_revpct;
    vector<DataPoint> m_dataPoints;
    int m_dim; // feature dimension: only for data
    int m_nbOfClasses; // number of classes: only for label

public:

    DataLoader(string dataSetPath, bool trainFlag, const string& kind, const string& revpct="0");

    inline const DataPoint& getDataPoint(int j) const { return m_dataPoints[j]; }

    int size() const { return m_dataPoints.size(); }

    int getDim() const { return m_dim; }

    int getNbOfClasses() const { return m_nbOfClasses; }
    
    string getDataSetName() const { return m_dataSetName; }

    vector<float> loadFileFloat(string filePath);

    vector<int> loadFileInt(string filePath);

    void prepareFeatureData(vector<int> dataPoint, vector<int> dataDim, vector<float> dataValue);

    void prepareLabelData(vector<int> labelPoint, vector<int> labelDim);
};