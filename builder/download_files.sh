echo "Creating collection and duplicates directory..."
mkdir -p /shared/topics

echo "Downloading benchmarks"
wget -c https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json -P /shared/topics
wget -c https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json -P /shared/topics
wget -c https://raw.githubusercontent.com/daltonj/treccastweb/master/2021/2021_manual_evaluation_topics_v1.0.json -P /shared/topics
wget -c https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/training/train_topics_v1.0.json -P /shared/topics