import re


def clean_sample_char(text):
    """
    Function to clean the file name ( pre-processor).
    :param text: the file name
    :return: the text preprocessed
    """

    if '.' in text:
        text = '.'.join(text.split('.')[:-1])

    ortho_code_num = re.sub(r'[^A-Za-z]+', '0', text)
    ortho_code_num = re.sub(r'[a-zA-Z]+', '1', ortho_code_num)

    ortho_code = re.sub(r'[A-Z]', 'C', text)
    ortho_code = re.sub(r'[a-z]', 'c', ortho_code)
    ortho_code = re.sub(r'[0-9]', 'N', ortho_code)
    ortho_code = re.sub(r'[^a-zA-Z0-9]', 'P', ortho_code)

    clean_text = re.sub(r'[0-9]+', '$', text)
    clean_text = re.sub(r'[^$0-9A-Za-z\s]+', '#', clean_text)
    clean_text = re.sub(r'(?:^| )\w(?:$| )', '', clean_text)
    clean_text = clean_text.strip()

    return clean_text + ' ' + ortho_code_num + ' ' + ortho_code
