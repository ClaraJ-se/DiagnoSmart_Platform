import os

def get_path():
    
    paths = {}

    paths['data'] = './data'
    paths['raw'] = os.path.join(paths['data'],"raw/mimic-iv-ed-2.2")
    paths['raw_ed'] = os.path.join(paths['raw'],"ed")
    paths['wrangling'] = os.path.join(paths['data'], "wrangling")
    paths['preprocessing'] = os.path.join(paths['data'], "preprocessing")
    
    paths['selection'] = os.path.join(paths['data'], "selection")

    paths['training'] = os.path.join(paths['data'], "training")

    for value in paths.values():
        os.makedirs(value, exist_ok=True)
        
    return paths


    



