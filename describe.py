import pandas as pd
import os
import io
from tabulate import tabulate
import logging

def process_csv_files(input_folder):
    def infer_and_rename_dtype(series):
        dtype = pd.api.types.infer_dtype(series)
        if dtype in ['integer', 'mixed-integer']:
            return 'Integer'
        elif dtype in ['floating', 'mixed-integer-float']:
            return 'Float'
        elif dtype in ['string', 'unicode', 'bytes']:
            return 'String'
        elif dtype == 'boolean':
            return 'Boolean'
        elif dtype == 'datetime':
            return 'Datetime'
        elif dtype in ['timedelta']:
            return 'Timedelta'
        elif dtype == 'complex':
            return 'Complex'
        else:
            return 'Unknown'

    output_excel_path = os.path.join(input_folder, 'synthetic_tables.xlsx')

    try:
        with pd.ExcelWriter(output_excel_path) as writer:
            for file_name in os.listdir(input_folder):
                if file_name.endswith('.csv'):
                    file_path = os.path.join(input_folder, file_name)
                    df = pd.read_csv(file_path)
                    synthetic_table = pd.DataFrame({
                        'Column Name': df.columns,
                        'Data Type': [infer_and_rename_dtype(df[col]) for col in df.columns],
                        'Description': [''] * len(df.columns)
                    })
                    sheet_name = os.path.splitext(file_name)[0]
                    synthetic_table.to_excel(writer, sheet_name=sheet_name, index=False)
        logging.info(f"Synthetic tables have been written to {output_excel_path}.")
    except Exception as e:
        logging.error(f"Error processing CSV files: {e}")

class DataFrameDescriber:
    def __init__(self, dataframe):
        self.df = dataframe
        logging.info("DataFrameDescriber initialized.")

    def print_formatted(self, data, title):
        """Prints data in a table format with a title."""
        print(title)
        print(tabulate(data, headers='keys', tablefmt='psql'))
        print("\n")

    def get_info(self):
        """Print formatted information about DataFrame including the index dtype and columns, non-null values and memory usage."""
        buf = io.StringIO()
        self.df.info(buf=buf)
        info = buf.getvalue()
        logging.info("DataFrame Information:\n" + info)

    def get_description(self, include='all'):
        """Generate and print formatted descriptive statistics."""
        description = self.df.describe(include=include)
        self.print_formatted(description, "Descriptive Statistics:")
        logging.info("Descriptive statistics generated.")

    def get_missing_values(self):
        """Print formatted count of missing values per column."""
        missing_values = self.df.isnull().sum()
        self.print_formatted(missing_values.to_frame('Missing Values'), "Missing Values Count per Column:")
        logging.info("Missing values count per column generated.")

    def get_missing_values_percentage_by_column(self):
        """Print formatted percentage of missing values per column."""
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        self.print_formatted(missing_percentage.to_frame('Missing Values Percentage'), "Missing Values Percentage per Column:")
        logging.info("Missing values percentage per column generated.")

    def get_missing_values_percentage_by_row(self):
        """Print formatted percentage of missing values per row."""
        missing_percentage_rows = (self.df.isnull().sum(axis=1) / self.df.shape[1]) * 100
        self.print_formatted(missing_percentage_rows.describe().to_frame('Missing Values Percentage'), "Missing Values Percentage per Row Statistics:")
        logging.info("Missing values percentage per row generated.")

    def get_unique_values(self):
        """Print formatted number of unique values per column."""
        unique_values = self.df.nunique()
        self.print_formatted(unique_values.to_frame('Unique Values'), "Unique Values per Column:")
        logging.info("Unique values per column generated.")

    def get_correlation(self):
        """Print formatted correlation matrix of the DataFrame."""
        correlation = self.df.corr()
        self.print_formatted(correlation, "Correlation Matrix:")
        logging.info("Correlation matrix generated.")

    def export_to_excel(self, output_path):
        """Export all descriptions and statistics to an Excel file."""
        try:
            with pd.ExcelWriter(output_path) as writer:
                self.df.describe(include='all').to_excel(writer, sheet_name='Descriptive Statistics')
                self.df.isnull().sum().to_frame('Missing Values Count').to_excel(writer, sheet_name='Missing Values Count')
                ((self.df.isnull().sum() / len(self.df)) * 100).to_frame('Missing Values Percentage').to_excel(writer, sheet_name='Missing Values Percentage')
                self.df.nunique().to_frame('Unique Values').to_excel(writer, sheet_name='Unique Values')
            logging.info(f"Report exported to {output_path}")
        except Exception as e:
            logging.error(f"Error exporting report to Excel: {e}")
