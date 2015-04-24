// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <boost/algorithm/string.hpp>  // NOLINT(legal/copyright)
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");

void read_feature_and_label_form_string_or_die(
    string line, int channels, Datum* feature, int* label) {
  // buff
  //LOG(INFO) << line;
  vector<string> cells, indval;

  // init feature datum
  feature->set_channels(channels);
  feature->set_height(1);
  feature->set_width(1);
  feature->clear_data();
  feature->clear_float_data();
  feature->mutable_float_data()->Reserve(channels);
  for (int s = 0; s < channels; s++) {
    feature->add_float_data(0.f);
  }

  // split lines
  boost::split(cells, line, boost::is_any_of(" \t"));
  *label = boost::lexical_cast<int>(cells[0]);
  feature->set_label(*label);

  // Parse sparse format features
  float* pfeat = feature->mutable_float_data()->mutable_data();
  int i = 0;
  BOOST_FOREACH(string cell, cells) {
    if (i++ == 0) { continue; }
    boost::split(indval, cell, boost::is_any_of(":"));
    CHECK_EQ(indval.size(), 2);
    unsigned int ind = boost::lexical_cast<unsigned int>(indval[0]);
    float val = boost::lexical_cast<float>(indval[1]);
    pfeat[ind] = val;
  }
  return;
}

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of svm_data to the leveldb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_svm_data NUM_FEATURE SVM_FORMAT_FILE DB_FILE\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }


  std::ifstream infile(argv[2]);
  std::vector<std::string > lines;
  std::string filename;

  string line;
  while (std::getline(infile, line)) {
    // trim spaces
    boost::trim(line);
    // skip empty lines
    if (line.empty()) {
      continue;
    }
    lines.push_back(line);
  }

  if (FLAGS_shuffle) {
      LOG(ERROR) << "Shuffling data";
      shuffle(lines.begin(), lines.end());
  }
  else {
      LOG(ERROR) << "NO shuffle";
  }
  LOG(ERROR) << "A total of " << lines.size() << " instances.";

  int num_channel = atoi(argv[1]);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB("leveldb"));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[3]);
  Datum datum;
  int label;
  int count = 0;
  const int kMaxKeyLength = 10240;
  char key_cstr[kMaxKeyLength];

  for (int line_id = 0; line_id < lines.size(); ++line_id) {

    read_feature_and_label_form_string_or_die(lines[line_id], num_channel, &datum, &label);

    // sequential
    int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id,
        lines[line_id].c_str());

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(string(key_cstr, length), out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(ERROR) << "Processed " << count << " instances.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(ERROR) << "Processed " << count << " instances.";
  }
  return 0;
}
