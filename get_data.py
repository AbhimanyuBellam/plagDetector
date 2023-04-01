import pandas as pd 
from summarizer import BartModel
from tqdm import tqdm

class GenerateSummaries():
    def __init__(self, input_file = "GPT-wiki-intro.csv"):
        self.summarizer = BartModel()
        wiki_data = pd.read_csv(input_file)
        # print(wiki_data.columns)
        self.ai_generated_data = wiki_data["generated_intro"]
        self.human_data = wiki_data["wiki_intro"]
        self.map_dict = {"AI": self.ai_generated_data, "H":self.human_data}
        self.write_dict = {"AI": "ai_text_summaries", "H": "human_text_summaries"}


    def generate_summary(self, start_end_indexes, type_ = "AI"):
        input_data = self.map_dict[type_]
        start_index, end_index = start_end_indexes

        save_path = f"{self.write_dict[type_]}_{start_index}_{end_index}.csv"

        summaries = []

        for i in tqdm(range(start_index, end_index)):
            text = input_data[i]
            # print (text)
            summary = self.summarizer.get_summary(text)
            summaries.append(summary)

        df = pd.DataFrame()
        df["Summary No."] = [i+1 for i in range(start_index, end_index)]
        df["H_Summaries"] = summaries

        df.to_csv(save_path, index=False)

        return 


if __name__ == "__main__":
    summary_generator = GenerateSummaries()
    summary_generator.generate_summary([20000, 40000], type_="H")



# print (ai_generated_data[5])
# print (human_data[9])









