import numpy as np

"""
Fast normalization for entire matrix, turning everything into a string and replacing anomalies
"""
def fast_normalize(X):
    out = X.astype(str)

    mask = (out == "nan") | (out == "") | (out == "None")
    out[mask] = "MISSING"

    return out

def is_real_number_array(X):
    if X.dtype.kind in ('f', 'c'):
        return True

    if X.dtype.kind != 'O':
        return False
    
    # Fast float detection
    # Flatten array into list
    for v in X.ravel():
        try:
            # Check if its a float
            if isinstance(v, (float, np.floating)):
                if not v.is_integer():
                    return True
            # Check if its pretending to be a string
            elif isinstance(v, str):
                    f = float(v)
                    if not f.is_integer():
                        return True
        except (ValueError,TypeError):
            pass
    return False

if __name__ == "__main__":
    print(fast_normalize(np.array([[0,1],[1,0]])))





