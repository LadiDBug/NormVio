import os
import json


def load_data(path):
    with open(path, 'r', encoding='utf-8') as fr:
        return [json.loads(line) for line in fr]

def save_data(path, data):
    with open(path, 'w', encoding='utf-8') as fw:
        for comment in data:
            json.dump(comment, fw)
            fw.write('\n')

def main():
    dataset_dir  = "C:/Users/Diana/Desktop/dataset"
    new_dataset_dir = "C:/Users/Diana/Desktop/new_NormVio_dataset"
    
    for file in os.listdir(dataset_dir):

        path_data = os.path.join(dataset_dir, file)
        data = load_data(path_data)

        new_data = []
        for comment in data: 
            rule_texts = comment["rule_texts"]
            if " ||| " not in rule_texts:
                new_data.append(comment)
        
        output_path = os.path.join(new_dataset_dir, file)
        save_data(output_path, new_data)

if __name__ == "__main__":
    main()
