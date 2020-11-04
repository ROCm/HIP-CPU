/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#ifndef RESULT_DATABASE_H
#define RESULT_DATABASE_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <cfloat>


// ****************************************************************************
// Class:  ResultDatabase
//
// Purpose:
//   Track numerical results as they are generated.
//   Print statistics of raw results.
//
// Programmer:  Jeremy Meredith
// Creation:    June 12, 2009
//
// Modifications:
//    Jeremy Meredith, Wed Nov 10 14:20:47 EST 2010
//    Split timing reports into detailed and summary.  E.g. for serial code,
//    we might report all trial values, but skip them in parallel.
//
//    Jeremy Meredith, Thu Nov 11 11:40:18 EST 2010
//    Added check for missing value tag.
//
//    Jeremy Meredith, Mon Nov 22 13:37:10 EST 2010
//    Added percentile statistic.
//
//    Jeremy Meredith, Fri Dec  3 16:30:31 EST 2010
//    Added a method to extract a subset of results based on test name.  Also,
//    the Result class is now public, so that clients can use them directly.
//    Added a GetResults method as well, and made several functions const.
//
// ****************************************************************************
class ResultDatabase {
   public:
    //
    // A performance result for a single SHOC benchmark run.
    //
    struct Result {
        std::string test;           // e.g. "readback"
        std::string atts;           // e.g. "pagelocked 4k^2"
        std::string unit;           // e.g. "MB/sec"
        std::vector<double> value;  // e.g. "837.14"
        double GetMin() const;
        double GetMax() const;
        double GetMedian() const;
        double GetPercentile(double q) const;
        double GetMean() const;
        double GetStdDev() const;

        bool operator<(const Result& rhs) const;

        bool HadAnyFLTMAXValues() const {
            for (decltype(value.size()) i = 0u; i < value.size(); ++i) {
                if (value[i] >= FLT_MAX) return true;
            }
            return false;
        }
    };

   protected:
    std::vector<Result> results;

   public:
    void AddResult(const std::string& test, const std::string& atts,
                   const std::string& unit, double value);
    void AddResults(const std::string& test, const std::string& atts,
                    const std::string& unit, const std::vector<double>& values);
    std::vector<Result> GetResultsForTest(const std::string& test);
    const std::vector<Result>& GetResults() const;
    void ClearAllResults();
    void DumpDetailed(std::ostream&);
    void DumpSummary(std::ostream&);
    void DumpCsv(std::string fileName);

   private:
    bool IsFileEmpty(std::string fileName);
};


#endif