#include "json_parser.h"
#include "event_counter.h"

#include <cassert>
#include <cctype>
#ifndef _MSC_VER
#include <dirent.h>
#include <unistd.h>
#endif
#include <cinttypes>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "linux-perf-events.h"
#ifdef __linux__
#include <libgen.h>
#endif
//#define DEBUG
#include "simdjson/common_defs.h"
#include "simdjson/isadetection.h"
#include "simdjson/jsonioutil.h"
#include "simdjson/jsonparser.h"
#include "simdjson/parsedjson.h"
#include "simdjson/stage1_find_marks.h"
#include "simdjson/stage2_build_tape.h"

#include <functional>

#include "benchmarker.h"

using namespace simdjson;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;
using std::ostream;
using std::ofstream;
using std::exception;

// Stash the exe_name in main() for functions to use
char* exe_name;

void print_usage(ostream& out) {
  out << "Usage: " << exe_name << " [-v] [-n #] [-s STAGE] [-a ARCH] <jsonfile> ..." << endl;
  out << endl;
  out << "Runs the parser against the given json files in a loop, measuring speed and other statistics." << endl;
  out << endl;
  out << "Options:" << endl;
  out << endl;
  out << "-n #       - Number of iterations per file. Default: 200" << endl;
  out << "-v         - Verbose output." << endl;
  out << "-s STAGE   - Stop after the given stage." << endl;
  out << "             -s stage1 - Stop after find_structural_bits." << endl;
  out << "             -s all    - Run all stages." << endl;
  out << "-a ARCH    - Use the parser with the designated architecture (HASWELL, WESTMERE" << endl;
  out << "             or ARM64). By default, detects best supported architecture." << endl;
}

void exit_usage(string message) {
  cerr << message << endl;
  cerr << endl;
  print_usage(cerr);
  exit(EXIT_FAILURE);
}

struct option_struct {
  Architecture architecture = Architecture::UNSUPPORTED;
  bool stage1_only = false;

  int32_t iterations = 400;

  bool verbose = false;

  option_struct(int argc, char **argv) {
    #ifndef _MSC_VER
      int c;

      while ((c = getopt(argc, argv, "vtn:a:s:")) != -1) {
        switch (c) {
        case 'n':
          iterations = atoi(optarg);
          break;
        case 'v':
          verbose = true;
          break;
        case 'a':
          architecture = parse_architecture(optarg);
          if (architecture == Architecture::UNSUPPORTED) {
            exit_usage(string("Unsupported option value -a ") + optarg + ": expected -a HASWELL, WESTMERE or ARM64");
          }
          break;
        case 's':
          if (!strcmp(optarg, "stage1")) {
            stage1_only = true;
          } else if (!strcmp(optarg, "all")) {
            stage1_only = false;
          } else {
            exit_usage(string("Unsupported option value -s ") + optarg + ": expected -s stage1 or all");
          }
          break;
        default:
          exit_error("Unexpected argument " + c);
        }
      }
    #else
      int optind = 1;
    #endif

    // If architecture is not specified, pick the best supported architecture by default
    if (architecture == Architecture::UNSUPPORTED) {
      architecture = find_best_supported_architecture();
    }
  }
};

double diff(const benchmarker& feature, const benchmarker& baseline) {
  return (feature.stage1.best.elapsed_ns() - baseline.stage1.best.elapsed_ns()) / baseline.stats->blocks;
}
double diff_flip(const benchmarker& feature, const benchmarker& baseline) {
  // There are roughly 2650 branch mispredicts, so we have to scale it so it represents a per block amount
  return diff(feature, baseline) * 10000.0 / 2650.0;
}

int main(int argc, char *argv[]) {
  // Read options
  exe_name = argv[0];
  option_struct options(argc, argv);
  if (options.verbose) {
    verbose_stream = &cout;
  }

  // Initialize the event collector. We put this early so if it prints an error message, it's the
  // first thing printed.
  event_collector collector;

  // Set up benchmarkers by reading all files
  json_parser parser(options.architecture);

  benchmarker baseline           ("jsonexamples/baseline-0-structurals.json", parser, collector);
  benchmarker utf8               ("jsonexamples/baseline-utf-8.json", parser, collector);
  benchmarker utf8_half          ("jsonexamples/baseline-utf-8-half.json", parser, collector);
  benchmarker utf8_flip          ("jsonexamples/baseline-utf-8-half-flip.json", parser, collector);
  vector<char*> structural_filenames, structural_filenames_half, structural_filenames_flip;
  vector<benchmarker*> structurals, structurals_half, structurals_flip;
  for (size_t i=1; i<=23; i++) {
    char* filename = (char*)malloc(200);
    sprintf(filename, "jsonexamples/baseline-%lu-structurals.json", i);
    structurals.push_back(new benchmarker(filename, parser, collector));
    structural_filenames.push_back(filename);

    filename = (char*)malloc(200);
    sprintf(filename, "jsonexamples/baseline-%lu-structurals-half.json", i);
    structurals_half.push_back(new benchmarker(filename, parser, collector));
    structural_filenames_half.push_back(filename);

    filename = (char*)malloc(200);
    sprintf(filename, "jsonexamples/baseline-%lu-structurals-half-flip.json", i);
    structurals_flip.push_back(new benchmarker(filename, parser, collector));
    structural_filenames_flip.push_back(filename);
  }

  // Run the benchmarks
  progress_bar progress(options.iterations, 50);
  for (int iteration = 0; iteration < options.iterations; iteration++) {
    if (!options.verbose) { progress.print(iteration); }
    // Benchmark each file once per iteration
    baseline.run_iteration(options.stage1_only);
    utf8.run_iteration(options.stage1_only);
    utf8_half.run_iteration(options.stage1_only);
    utf8_flip.run_iteration(options.stage1_only);
    for (size_t i=0;i<structurals.size();i++) {
      structurals[i]->run_iteration(options.stage1_only);
      structurals_half[i]->run_iteration(options.stage1_only);
      structurals_flip[i]->run_iteration(options.stage1_only);
    }
  }
  if (!options.verbose) { progress.erase(); }
    

  printf("baseline (ns/block)");
  printf(",utf-8");
  for (size_t i=0;i<structurals.size(); i++) { printf(",%lu structurals", i+1); }
  printf(",utf-8 branch miss");
  for (size_t i=0;i<structurals.size(); i++) { printf(",%lu structurals branch miss", i+1); }
  printf("\n");
  printf("%f", baseline.stage1.best.elapsed_ns() / baseline.stats->blocks);
  printf(",%f", diff(utf8, baseline));
  printf(",%f", diff(*structurals[0], baseline));
  for (size_t i=1;i<structurals.size(); i++) {
    printf(",%f", diff(*structurals[i], *structurals[i-1]));
  }
  printf(",%f", diff_flip(utf8_flip, utf8_half));
  for (size_t i=0;i<structurals.size(); i++) {
    printf(",%f", diff_flip(*structurals_flip[i], *structurals_half[i]));
  }
  printf("\n");

  for (size_t i=0; i<structurals.size(); i++) {
    free(structural_filenames[i]);
    free(structural_filenames_half[i]);
    free(structural_filenames_flip[i]);
    delete structurals[i];
    delete structurals_half[i];
    delete structurals_flip[i];
  }

  return EXIT_SUCCESS;
}
