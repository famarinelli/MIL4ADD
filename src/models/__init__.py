import importlib
import sys

def get_model_class(model_name: str):
    """
    Importa dinamicamente una classe modello da src/models/{model_name}.py.
    
    Args:
        model_name (str): Il nome del file (senza .py) e della classe. 
                          Es: 'MILModel' cercherà src/models/MILModel.py e la classe MILModel.
    
    Returns:
        class: La classe del modello (non istanziata).
    """
    try:
        # Costruiamo il percorso del modulo: src.models.NomeModello
        module_path = f"src.models.{model_name}"
        
        # Importiamo il modulo
        module = importlib.import_module(module_path)
        
        # Recuperiamo la classe con lo stesso nome del file
        model_class = getattr(module, model_name)
        
        return model_class
        
    except ImportError as e:
        raise ImportError(f"Impossibile trovare il modulo '{model_name}' in src/models/. "
                          f"Assicurati che il file esista: src/models/{model_name}.py") from e
    except AttributeError as e:
        raise AttributeError(f"Il modulo '{model_name}' è stato caricato, ma non contiene la classe '{model_name}'.") from e