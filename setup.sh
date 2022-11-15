echo "Downloading datasets..."


echo "Downloading CAsT Y4..."
wget -c https://raw.githubusercontent.com/daltonj/treccastweb/master/2022/2022_evaluation_topics_flattened_duplicated_v1.0.json -P data/cast/year_4
wget -c https://raw.githubusercontent.com/daltonj/treccastweb/master/2022/cast2022.qrel -P data/cast/year_4


echo "Downloading Clariq..."
wget -c https://raw.githubusercontent.com/aliannejadi/ClariQ/master/data/dev.tsv -P data/clariq
wget -c https://raw.githubusercontent.com/aliannejadi/ClariQ/master/data/train.tsv -P data/clariq


echo "Datasets downloaded!"