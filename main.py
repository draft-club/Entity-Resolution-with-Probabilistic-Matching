import os
import pandas as pd
from prepare import DataFrameProcessor
from mapping import DataFrameMapper
from describe import DataFrameDescriber
from normalize import AddressNormalization
from psig_model import PsigModel
from constants import INPUT_FOLDER, OUTPUT_FOLDER, DICTIONARY_FOLDER

def process_csv_files(input_folder):
    output_excel_path = os.path.join(input_folder, 'synthetic_tables.xlsx')
    with pd.ExcelWriter(output_excel_path) as writer:
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.csv'):
                file_path = os.path.join(input_folder, file_name)
                df = pd.read_csv(file_path)
                synthetic_table = pd.DataFrame({
                    'Column Name': df.columns,
                    'Data Type': [pd.api.types.infer_dtype(df[col]) for col in df.columns],
                    'Description': [''] * len(df.columns)
                })
                sheet_name = os.path.splitext(file_name)[0]
                synthetic_table.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Synthetic tables have been written to {output_excel_path}.")

def main():
    input_path = os.path.join(INPUT_FOLDER, 'Scholar_2.csv')
    dict_path = os.path.join(DICTIONARY_FOLDER, 'mapping_dts.json')
    output_path = os.path.join(OUTPUT_FOLDER, 'output.csv')

    # Prepare Data
    processor = DataFrameProcessor(input_path)
    processor.drop_columns_with_high_nas()

    # Mapping
    """
    mapper = DataFrameMapper(processor.df, dict_path)
    mapper.apply_mapping()
    mapper.filter_columns()
    
    """

    mapper = processor

    # Describe
    describer = DataFrameDescriber(mapper.df)
    describer.get_info()
    describer.get_description()
    describer.get_missing_values()
    describer.get_missing_values_percentage_by_column()
    describer.get_missing_values_percentage_by_row()
    describer.get_unique_values()

    # Normalize
    '''
    normalizer = AddressNormalization(mapper.df)
    normalizer.normalize_addresses('adresse')
    normalizer.export_results(output_path)
    
    '''

    normalizer = describer
    # Psig Model
    psig_model = PsigModel(normalizer.df)
    components = psig_model.run()

    # Output the results of the Psig model
    output_psig_path = os.path.join(OUTPUT_FOLDER, 'psig_model_output.csv')
    component_df = pd.DataFrame([(key, list(val)) for key, val in components.items()], columns=['Component_ID', 'Records'])
    component_df.to_csv(output_psig_path, index=False)
    print(f"Psig model results saved to {output_psig_path}")

if __name__ == '__main__':
    main()
