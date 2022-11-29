import json
import csv


if __name__ == "__main__":

    annotation_dict = {}

    with open("data/cast/year_4/2022_evaluation_topics_flattened_duplicated_v1.0.json") as f:
        topics = json.load(f)

    with open("data/cast/year_4/Year 4 Topic Annotations - User Query Annotations.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for line_number, row in enumerate(csv_reader):
            if line_number == 0:
                continue
            else:
                turn_id = f"{row[0]}_{row[2]}-{row[3]}"
                information_need = row[6]
                utterance_type = row[10]

                annotation_dict[turn_id] = {
                    "information_need": information_need,
                    "utterance_type": utterance_type
                }

    for topic in topics:
        for turn in topic['turn']:
            turn_id = f"{topic['number']}_{turn['number']}"
            turn["information_need"] = annotation_dict[turn_id]["information_need"]
            turn["utterance_type"] = annotation_dict[turn_id]["utterance_type"]


    with open("data/cast/year_4/annotated_topics.json", "w") as annotated_f:
        json.dump(topics, annotated_f, indent=4, ensure_ascii=True)
