import argparse 
import pathlib
import pdb 
import json 
from collections import defaultdict
from typing import List, Dict, Tuple

class Index:
    def __init__(self, 
                train_path: pathlib.Path,
                summary_stat = max):
        self.train_path = train_path
        
        (self.action_to_summary_lut, 
        self.summary_to_action_lut, 
        self.per_task_summary_vote_lut) = self.load_from_path()

        self.per_task_summary_vote_lut = self.normalize_task_summary_vote_lut(self.per_task_summary_vote_lut, summary_stat)

    def load_from_path(self) -> Tuple[Dict[Tuple[str], str], Dict[str, Tuple[str]], Dict[int, Dict[str, int]]]:
        action_to_summary_lut = {}
        summary_to_action_lut = {}
        per_task_summary_vote_lut = defaultdict(lambda: defaultdict(int))
        for file in self.train_path.glob("*.json"):
            with open(file) as f:
                data = json.load(f)
            alignment = data['action_alignment']
            if alignment is None:
                continue
            task = int(data['task_id'].split("_")[0])
            for step in alignment:
                summary = step['summary_action_str']
                per_task_summary_vote_lut[task][summary] += 1
                actions = tuple(step['robot_actions_str'].split(", "))
                action_to_summary_lut[actions] = summary
                summary_to_action_lut[summary] = actions

        return (action_to_summary_lut, summary_to_action_lut, per_task_summary_vote_lut) 

    def normalize_task_summary_vote_lut(self, 
                per_task_summary_vote_lut: Dict[int, Dict[str, int]],
                summary_stat = max) -> Dict[int, Dict[str, int]]:
        """
        Normalize the summary vote count by task 
        """
        for task, summary_to_vote_lut in per_task_summary_vote_lut.items():
            total = summary_stat(summary_to_vote_lut.values())
            for summary, vote in summary_to_vote_lut.items():
                summary_to_vote_lut[summary] = vote / total
            per_task_summary_vote_lut[task] = summary_to_vote_lut
        return per_task_summary_vote_lut

    def get_vote_count(self, 
                       task: int, 
                       action_seq: List[str]):

        print(f"action seq: {action_seq}")
        pdb.set_trace()
        summary = self.action_to_summary_lut[action_seq]
        count = 0
        if task is None:
            for task in self.per_task_summary_vote_lut:
                count += self.per_task_summary_vote_lut[task][summary]
        else:
            count = self.per_task_summary_vote_lut[task][summary]
        return count 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--summary_stat", default="max")
    args = parser.parse_args()

    if args.summary_stat == "max":
        summary_stat = max
    else:
        summary_stat = sum
    index = Index(pathlib.Path(args.train_path), summary_stat=summary_stat)
    pdb.set_trace()