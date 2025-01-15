import sys
from pathlib import Path

# percorso di ricerca dei moduli in 'src'
src_path = Path(__file__).resolve().parent / "src"
sys.path.append(str(src_path))

from src import knowledge_base_logic, supervised_learning, probability_learning, unsupervised_learning, bayesian_network

def execute_main(scelta):
    if scelta == "knowledge_base_logic":
        knowledge_base_logic.main()
    elif scelta == "supervised_learning":
        supervised_learning.main()
    elif scelta == "probability_learning":
        probability_learning.main()
    elif scelta == "unsupervised_learning":
        unsupervised_learning.main()
    elif scelta == "bayesian_network":
        bayesian_network.main()
    else:
        print(f"Errore: scelta '{scelta}' non valida.")

if __name__ == "__main__":
    print("Scegli quale 'main' eseguire:")
    print("1 - Ragionamento logico su KB")
    print("2 - Appprendimento supervisionato")
    print("3 - Apprendimento probabilistico su NB")
    print("4 - Apprendimento non supervisionato")
    print("5 - Bayesian Network")
    scelta = input("Inserisci il numero della tua scelta: ")

    elenco_scelta = {
        "1": "knowledge_base_logic",
        "2": "supervised_learning",
        "3": "probability_learning",
        "4": "unsupervised_learning",
        "5": "bayesian_network",
    }

    seleziona_modulo = elenco_scelta.get(scelta)
    if seleziona_modulo:
        execute_main(seleziona_modulo)
    else:
        print("Scelta non valida. Programma terminato.")
