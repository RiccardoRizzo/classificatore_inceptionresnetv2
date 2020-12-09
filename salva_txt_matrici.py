import numpy as np

def matrice_a_stringa(matrice, stringa="", str_join_righe = ", "):
    """
    Trasforma la matrice in una stringa csv
    """
    riga = []
    mm = list(matrice)
    for r in mm:
        riga.append( str_join_righe.join([str(num) for num in list(r) ]) )
    out = "\n".join(riga)
    if len(stringa) > 0:
        stringa = "\n# " + stringa
        out.append(stringa)
    return out

def salva_matrice_csv(filename, matrice, modo = "w", stringa = ""):
    """
    Salva una matrice come file csv
    """
    mm_csv = matrice_a_stringa(matrice)
    with  open(filename, modo) as fout:
        fout.write(mm_csv)


