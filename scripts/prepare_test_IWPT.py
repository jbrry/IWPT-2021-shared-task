import sys
import trankit
from argparse import ArgumentParser

def argparser():
    """Arguments for Trankit."""
    ap = ArgumentParser()
    ap.add_argument('--train-txt', type=str, default='/home/jbarry/spinning-storage/jbarry/IWPT-2021-shared-task/data/train-dev/UD_English-EWT/en_ewt-ud-train.txt',
                    help='Train text file.')
    ap.add_argument('--train-conllu', type=str, default='/home/jbarry/spinning-storage/jbarry/IWPT-2021-shared-task/data/train-dev/UD_English-EWT/en_ewt-ud-train.conllu',
                    help='Train conllu file.')
    ap.add_argument('--dev-txt', type=str, default='/home/jbarry/spinning-storage/jbarry/IWPT-2021-shared-task/data/train-dev/UD_English-EWT/en_ewt-ud-dev.txt',
                    help='Dev text file.')
    ap.add_argument('--dev-conllu', type=str, default='/home/jbarry/spinning-storage/jbarry/IWPT-2021-shared-task/data/train-dev/UD_English-EWT/en_ewt-ud-dev.conllu',
                    help='Dev conllu file.')
    ap.add_argument('--category', type=str, default='customized', # customized-mwt
                    help='Trankit category.')
    ap.add_argument('--task', type=str, default='tokenize',
                    help='Trankit task.')
    ap.add_argument('--save-dir', type=str, default='./logs/trankit_models',
                    help='Directory for saving trained model.')
    ap.add_argument('--train', default=False, action='store_true',
                    help='Train model.')
    ap.add_argument('--predict', default=False, action='store_true',
                    help='Predict model.')
                   
    return ap

def main(argv):
    args = argparser().parse_args(argv[1:])

    if args.train:
        import trankit

        # initialize a trainer for the task
        trainer = trankit.TPipeline(
            training_config={
            'category': args.category,
            'task': args.task,
            'save_dir': args.save_dir,
            'train_txt_fpath': args.train_txt,
            'train_conllu_fpath': args.train_conllu,
            'dev_txt_fpath': args.dev_txt,
            'dev_conllu_fpath': args.dev_conllu
            }
        )

        # start training
        trainer.train()
    
    if args.predict:
        # First verify the model
        trankit.verify_customized_pipeline(
            category=args.category,
            save_dir=args.save_dir,
        )

        from trankit import Pipeline
        p = Pipeline(lang=args.category, cache_dir=args.save_dir)

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))














