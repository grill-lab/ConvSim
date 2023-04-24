echo "Downloading datasets..."


echo "Downloading CAsT Y4..."
wget -c https://raw.githubusercontent.com/daltonj/treccastweb/master/2022/2022_evaluation_topics_flattened_duplicated_v1.0.json -P data/cast/year_4
wget -c https://raw.githubusercontent.com/paulowoicho/treccastweb/master/2022/cast2022.qrel -P data/cast/year_4
wget -c https://raw.githubusercontent.com/daltonj/treccastweb/master/2022/2022_mixed_initiative_question_pool.json -P data/cast/year_4
wget -c https://raw.githubusercontent.com/paulowoicho/treccastweb/master/2022/annotated_topics.json -P data/cast/year_4

# download index subset, extract, and delete
echo "Downloading a subset of the CAsT Y4 index..."
wget -c https://cast-y4-collection.s3.amazonaws.com/index_subset.tar.gz -P data/cast/year_4/indexes
tar -xvzf data/cast/year_4/indexes/index_subset.tar.gz -C data/cast/year_4/indexes
rm -rf data/cast/year_4/indexes/index_subset.tar.gz


echo "Downloading Clariq..."
wget -c https://raw.githubusercontent.com/aliannejadi/ClariQ/master/data/dev.tsv -P data/clariq
wget -c https://raw.githubusercontent.com/aliannejadi/ClariQ/master/data/train.tsv -P data/clariq


echo "Datasets downloaded!"
