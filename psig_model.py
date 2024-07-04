# psig_model.py

import pandas as pd
from collections import defaultdict
from itertools import combinations
import numpy as np
from constants import EXCEL_OUTPUT_PATH

class PsigModel:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the PsigModel with the given DataFrame.

        Args:
            df (pd.DataFrame): The input data.
        """
        self.df = df
        self.m_probabilities = {}
        self.u_probabilities = {}
        self.weights = {}
        self.signature_probabilities = {}
        self.inverted_index = defaultdict(set)

    def generate_signatures(self, record: str, min_length: int = 2, max_length: int = 3) -> set:
        """
        Generate n-gram signatures from the given record.

        Args:
            record (str): The input record.
            min_length (int): Minimum length of n-grams.
            max_length (int): Maximum length of n-grams.

        Returns:
            set: A set of n-gram signatures.
        """
        tokens = record.split()
        signatures = set()
        for length in range(min_length, max_length + 1):
            for i in range(len(tokens) - length + 1):
                signatures.add(' '.join(tokens[i:i + length]))
        return signatures

    def compute_m_u_probabilities(self):
        """
        Compute the m and u probabilities for each field.
        """
        fields = ['nom_prenom', 'date_naissance', 'adresse', 'ville', 'code_postal']
        total_records = len(self.df)

        for field in fields:
            field_values = self.df[field].fillna('')
            m_counts = defaultdict(int)
            u_counts = defaultdict(int)

            for value in field_values:
                m_counts[value] += 1

            for value in field_values:
                u_counts[value] += 1

            self.m_probabilities[field] = {k: m_counts[k] / total_records for k in m_counts}
            self.u_probabilities[field] = {k: (u_counts[k] / total_records) * 0.1 for k in u_counts}

        print("m_probabilities:", self.m_probabilities)
        print("u_probabilities:", self.u_probabilities)

    def compute_weights(self):
        """
        Compute the weights for each field based on m and u probabilities.
        """
        fields = ['nom_prenom', 'date_naissance', 'adresse', 'ville', 'code_postal']
        for field in fields:
            self.weights[field] = {}
            for value in self.m_probabilities[field]:
                m = self.m_probabilities[field][value]
                u = self.u_probabilities[field][value]
                self.weights[field][value] = {
                    'agree': np.log2(m / u) if u != 0 else 0,
                    'disagree': np.log2((1 - m) / (1 - u)) if u != 1 else 0
                }

        print("weights:", self.weights)

    def build_inverted_index(self, threshold: float = 0.1):
        """
        Build an inverted index for the signatures.

        Args:
            threshold (float): Threshold for filtering signatures based on probability.
        """
        for idx, record in enumerate(self.df['nom_prenom'].fillna('')):
            signatures = self.generate_signatures(record)
            print(f"Record {idx} signatures: {signatures}")  # Debugging line
            for sig in signatures:
                self.signature_probabilities[sig] = self.signature_probabilities.get(sig, 0) + 1

        total_signatures = sum(self.signature_probabilities.values())
        for sig in self.signature_probabilities:
            self.signature_probabilities[sig] /= total_signatures

        print("signature_probabilities:", self.signature_probabilities)  # Debugging line

        for idx, record in enumerate(self.df['nom_prenom'].fillna('')):
            signatures = self.generate_signatures(record)
            for sig in signatures:
                if self.signature_probabilities.get(sig, 0) > threshold:
                    self.inverted_index[sig].add(idx)

        print("inverted_index:", self.inverted_index)

    def generate_record_pairs(self):
        """
        Generate pairs of records based on the inverted index.
        """
        self.record_pairs = set()
        for sig, record_ids in self.inverted_index.items():
            if len(record_ids) > 1:
                for pair in combinations(record_ids, 2):
                    self.record_pairs.add(pair)

        print("record_pairs:", self.record_pairs)

    def filter_record_pairs(self, tau: float = 0.7):
        """
        Filter record pairs based on the linkage probability threshold.

        Args:
            tau (float): Linkage probability threshold.
        """
        self.filtered_pairs = set()
        for r1, r2 in self.record_pairs:
            signatures1 = self.generate_signatures(self.df['nom_prenom'].iloc[r1])
            signatures2 = self.generate_signatures(self.df['nom_prenom'].iloc[r2])
            common_signatures = signatures1.intersection(signatures2)
            if common_signatures:
                probabilities = [self.signature_probabilities[sig] for sig in common_signatures]
                linkage_prob = 1 - np.prod([1 - p for p in probabilities])
                if linkage_prob > tau:
                    self.filtered_pairs.add((r1, r2))

        print("filtered_pairs:", self.filtered_pairs)

    def calculate_match_weights(self):
        """
        Calculate the match weights for each pair of records.
        """
        self.match_weights = {}
        fields = ['nom_prenom', 'date_naissance', 'adresse', 'ville', 'code_postal']
        for r1, r2 in self.filtered_pairs:
            weight = 0
            for field in fields:
                value1 = self.df[field].iloc[r1]
                value2 = self.df[field].iloc[r2]
                if value1 == value2:
                    weight += self.weights[field][value1]['agree']
                else:
                    weight += self.weights[field][value1]['disagree']
            self.match_weights[(r1, r2)] = weight

        print("match_weights:", self.match_weights)

    def classify_pairs(self, upper_threshold: float, lower_threshold: float):
        """
        Classify pairs into matches, non-matches, and possible matches based on thresholds.

        Args:
            upper_threshold (float): Upper threshold for match weights.
            lower_threshold (float): Lower threshold for match weights.
        """
        self.matches = set()
        self.non_matches = set()
        self.possible_matches = set()

        for pair, weight in self.match_weights.items():
            if weight > upper_threshold:
                self.matches.add(pair)
            elif weight < lower_threshold:
                self.non_matches.add(pair)
            else:
                self.possible_matches.add(pair)

        print("matches:", self.matches)
        print("non_matches:", self.non_matches)
        print("possible_matches:", self.possible_matches)

    def export_results_to_excel(self, results: dict, output_path: str):
        """
        Export the matching results to an Excel file.

        Args:
            results (dict): The dictionary containing matches, non-matches, and possible matches.
            output_path (str): The path to save the Excel file.
        """
        with pd.ExcelWriter(output_path) as writer:
            for key, value in results.items():
                df = pd.DataFrame(list(value), columns=['Record 1', 'Record 2'])
                df.to_excel(writer, sheet_name=key, index=False)

            # Export m_probabilities
            m_prob_df = pd.DataFrame.from_dict(self.m_probabilities, orient='index').transpose()
            m_prob_df.to_excel(writer, sheet_name='m_probabilities')

            # Export u_probabilities
            u_prob_df = pd.DataFrame.from_dict(self.u_probabilities, orient='index').transpose()
            u_prob_df.to_excel(writer, sheet_name='u_probabilities')

            # Export signature probabilities
            sig_prob_df = pd.DataFrame(list(self.signature_probabilities.items()), columns=['Signature', 'Probability'])
            sig_prob_df.to_excel(writer, sheet_name='signature_probabilities')

            # Export inverted index
            inv_idx_df = pd.DataFrame([(k, v) for k, v in self.inverted_index.items()], columns=['Signature', 'Record Indices'])
            inv_idx_df.to_excel(writer, sheet_name='inverted_index')

    def run(self, upper_threshold: float, lower_threshold: float) -> dict:
        """
        Run the entire record linkage pipeline.

        Args:
            upper_threshold (float): Upper threshold for match weights.
            lower_threshold (float): Lower threshold for match weights.

        Returns:
            dict: A dictionary containing matches, non-matches, and possible matches.
        """
        self.compute_m_u_probabilities()
        self.compute_weights()
        self.build_inverted_index()
        self.generate_record_pairs()
        self.filter_record_pairs()
        self.calculate_match_weights()
        self.classify_pairs(upper_threshold, lower_threshold)
        results = {
            'matches': self.matches,
            'non_matches': self.non_matches,
            'possible_matches': self.possible_matches
        }
        self.export_results_to_excel(results, EXCEL_OUTPUT_PATH)
        return results

# Example usage
if __name__ == "__main__":
    data = {
        'nom_prenom': ['Jana Asher', 'Jane Asher', 'John Doe', 'Johnny Do', 'J. Doe', 'J. D.', 'Jana A.', 'J. Asher'],
        'date_naissance': ['1970-10-17', '1970-10-17', '1980-01-01', '1980-01-01', '1980-01-01', '1980-01-01', '1970-10-17', '1970-10-17'],
        'adresse': ['603 Brook Court', '1111 Jackson Ave', '123 Main St', '124 Main St', '123 Main St', '124 Main St', '603 Brook Ct', '603 Brook Ct'],
        'ville': ['CityA', 'CityA', 'CityB', 'CityB', 'CityB', 'CityB', 'CityA', 'CityA'],
        'code_postal': ['10001', '10001', '20002', '20002', '20002', '20002', '10001', '10001']
    }
    df = pd.DataFrame(data)
    model = PsigModel(df)
    results = model.run(upper_threshold=2, lower_threshold=0)
    print(results)
