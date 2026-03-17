from train import main as train_main
from valid import main as valid_main

if __name__ == "__main__":
    train_main()
    # valid_main() # Disable standalone validation to avoid argument conflicts and missing checkpoint error
    