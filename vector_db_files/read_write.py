import yaml

def read_file(file_path:str):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)  
    return data

def write_file(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as file:
            file.write(data)