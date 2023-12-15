import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='Training Dynamics Guided Knowledge Distillation.')

"""
Available datasets
- cll
- smk
- toxicity
- lung
- metabric-dr__200
- metabric-pam50__200
- tcga-2ysurvival__200
- tcga-tumor-grade__200
- prostate
"""

parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--dataset_size', type=int, help='100, 200, 330, 400, 800, 1600')
parser.add_argument('--dataset_feature_set', type=str, choices=['hallmark', '8000', '16000'], default='hallmark',
                    help='Note: implemented for Metabric only \
                        hallmark = 4160 common genes \
                        8000 = the 4160 common genes + 3840 random genes \
                        16000 = the 8000 genes above + 8000 random genes')


####### Model
parser.add_argument('--model', type=str, choices=['dnn', 'dietdnn', 'lasso', 'rf', 'lgb', 'tabnet', 'fsnet', 'cae', 'lassonet'], default='dnn')
parser.add_argument('--feature_extractor_dims', type=int, nargs='+', default=[100, 100, 10],  # use last dimnsion of 10 following the paper "Promises and perils of DKL" 
                    help='layer size for the feature extractor. If using a virtual layer,\
                            the first dimension must match it.')
parser.add_argument('--layers_for_hidden_representation', type=int, default=2, 
                    help='number of layers after which to output the hidden representation used as input to the decoder \
                            (e.g., if the layers are [100, 100, 10] and layers_for_hidden_representation=2, \
                            then the hidden representation will be the representation after the two layers [100, 100])')


parser.add_argument('--batchnorm', type=int, default=1, help='if 1, then add batchnorm layers in the main network. If 0, then dont add batchnorm layers')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate for the main network')
parser.add_argument('--gamma', type=float, default=0, 
                    help='The factor multiplied to the reconstruction error. \
                            If >0, then create a decoder with a reconstruction loss. \
                            If ==0, then dont create a decoder.')
parser.add_argument('--saved_checkpoint_name', type=str, help='name of the wandb artifact name (e.g., model-1dmvja9n:v0)')
parser.add_argument('--load_model_weights', action='store_true', dest='load_model_weights', help='True if loading model weights')
parser.set_defaults(load_model_weights=False)

####### Scikit-learn parameters
parser.add_argument('--lasso_C', type=float, default=1e3, help='lasso regularization parameter')

parser.add_argument('--rf_n_estimators', type=int, default=500, help='number of trees in the random forest')
parser.add_argument('--rf_max_depth', type=int, default=5, help='maximum depth of the tree')
parser.add_argument('--rf_min_samples_leaf', type=int, default=2, help='minimum number of samples in a leaf')

parser.add_argument('--lgb_learning_rate', type=float, default=0.1)
parser.add_argument('--lgb_max_depth', type=int, default=1)
parser.add_argument('--lgb_min_data_in_leaf', type=int, default=2)

parser.add_argument('--tabnet_lambda_sparse', type=float, default=1e-3, help='higher coefficient the sparser the feature selection')

parser.add_argument('--lassonet_lambda_start', default='auto', help='higher coefficient the sparser the feature selection')
parser.add_argument('--lassonet_gamma', type=float, default=0, help='higher coefficient the sparser the feature selection')
parser.add_argument('--lassonet_epochs', type=int, default=100)
parser.add_argument('--lassonet_M', type=float, default=10)

####### Sparsity
parser.add_argument('--sparsity_type', type=str, default=None,
                    choices=['global', 'local'], help="Use global or local sparsity")
parser.add_argument('--sparsity_method', type=str, default='sparsity_network',
                    choices=['learnable_vector', 'sparsity_network'], help="The method to induce sparsity")
parser.add_argument('--mixing_layer_size', type=int, help='size of the mixing layer in the sparsity network')
parser.add_argument('--mixing_layer_dropout', type=float, help='dropout rate for the mixing layer')

parser.add_argument('--sparsity_gene_embedding_type', type=str, default='nmf',
                    choices=['all_patients', 'nmf'], help='It`s applied over data preprocessed using `embedding_preprocessing`')
parser.add_argument('--sparsity_gene_embedding_size', type=int, default=50)
parser.add_argument('--sparsity_regularizer', type=str, default='L1',
                    choices=['L1', 'hoyer'])
parser.add_argument('--sparsity_regularizer_hyperparam', type=float, default=0,
                    help='The weight of the sparsity regularizer (used to compute total_loss)')


####### DKL
parser.add_argument('--grid_bound', type=float, default=5., help='The grid bound on the inducing points for the GP.')
parser.add_argument('--grid_size', type=int, default=64, help='Dimension of the grid of inducing points')


####### Weight predictor network
parser.add_argument('--wpn_embedding_type', type=str, default='nmf',
                    choices=['histogram', 'all_patients', 'nmf', 'svd'],
                    help='histogram = histogram x means (like FsNet)\
                            all_patients = randomly pick patients and use their gene expressions as the embedding\
                            It`s applied over data preprocessed using `embedding_preprocessing`')
parser.add_argument('--wpn_embedding_size', type=int, default=50, help='Size of the gene embedding')
parser.add_argument('--residual_embedding', type=str, default=None, choices=['resnet'],
                    help='Implement residual embeddings as e^* = e_{static} + f(e). This hyperparameter defines the type of function f')

parser.add_argument('--diet_network_dims', type=int, nargs='+', default=[100, 100, 100, 100],
                    help="None if you don't want a VirtualLayer. If you want a virtual layer, \
                            then provide a list of integers for the sized of the tiny network.")
parser.add_argument('--nonlinearity_weight_predictor', type=str, choices=['tanh', 'leakyrelu'], default='leakyrelu')
parser.add_argument('--softmax_diet_network', type=int, default=0, dest='softmax_diet_network',
                    help='If True, then perform softmax on the output of the tiny network.')

                        
####### Training
parser.add_argument('--use_best_hyperparams', action='store_true', dest='use_best_hyperparams',
                    help="True if you don't want to use the best hyperparams for a custom dataset")
parser.set_defaults(use_best_hyperparams=False)

parser.add_argument('--concrete_anneal_iterations', type=int, default=1000,
    help='number of iterations for annealing the Concrete radnom variables (in CAE and FsNet)')

parser.add_argument('--max_steps', type=int, default=10000, help='Specify the max number of steps to train.')
parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--patient_preprocessing', type=str, default='standard',
                    choices=['raw', 'standard', 'minmax'],
                    help='Preprocessing applied on each COLUMN of the N x D matrix, where a row contains all gene expressions of a patient.')
parser.add_argument('--embedding_preprocessing', type=str, default='minmax',
                    choices=['raw', 'standard', 'minmax'],
                    help='Preprocessing applied on each ROW of the D x N matrix, where a row contains all patient expressions for one gene.')


####### Training on the entire train + validation data
parser.add_argument('--train_on_full_data', action='store_true', dest='train_on_full_data', \
                    help='Train on the full data (train + validation), leaving only `--test_split` for testing.')
parser.set_defaults(train_on_full_data=False)
parser.add_argument('--path_steps_on_full_data', type=str, default=None, 
                    help='Path to the file which holds the number of steps to train.')



####### Validation
parser.add_argument('--metric_model_selection', type=str, default='cross_entropy_loss',
                    choices=['cross_entropy_loss', 'total_loss', 'balanced_accuracy'])

parser.add_argument('--patience_early_stopping', type=int, default=200,
                    help='Set number of checks (set by *val_check_interval*) to do early stopping.\
                            It will train for at least   args.val_check_interval * args.patience_early_stopping epochs')
parser.add_argument('--val_check_interval', type=int, default=5, 
                    help='number of steps at which to check the validation')

# type of data augmentation
parser.add_argument('--valid_aug_dropout_p', type=float, nargs="+", 
                    help="List of dropout data augmentation for the validation data loader.\
                            A new validation dataloader is created for each value.\
                            E.g., (1, 10) creates a dataloader with valid_aug_dropout_p=1, valid_aug_dropout_p=10\
                            in addition to the standard validation")
parser.add_argument('--valid_aug_times', type=int, nargs="+",
                    help="Number time to perform data augmentation on the validation sample.")


####### Testing
parser.add_argument('--testing_type', type=str, default='cross-validation',
                    choices=['cross-validation', 'fixed'],
                    help='`cross-validation` performs testing on the testing splits \
                            `fixed` performs testing on an external testing set supplied in a dedicated file')


####### Cross-validation
parser.add_argument('--repeat_id', type=int, default=0, help='each repeat_id gives a different random seed for shuffling the dataset')
parser.add_argument('--cv_folds', type=int, default=5, help="Number of CV splits")
parser.add_argument('--test_split', type=int, default=0, help="Index of the test split. It should be smaller than `cv_folds`")
parser.add_argument('--valid_percentage', type=float, default=0.1, help='Percentage of training data used for validation')
                            

####### Evaluation by taking random samples (with user-defined train/valid/test sizes) from the dataset
parser.add_argument('--evaluate_with_sampled_datasets', action='store_true', dest='evaluate_with_sampled_datasets')
parser.set_defaults(evaluate_with_sampled_datasets=False)
parser.add_argument('--custom_train_size', type=int, default=None)
parser.add_argument('--custom_valid_size', type=int, default=None)
parser.add_argument('--custom_test_size', type=int, default=None)


####### Optimization
parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw'], default='adamw')
parser.add_argument('--lr_scheduler', type=str, choices=['plateau', 'cosine_warm_restart', 'linear', 'lambda'], default='lambda')
parser.add_argument('--cosine_warm_restart_eta_min', type=float, default=1e-6)
parser.add_argument('--cosine_warm_restart_t_0', type=int, default=35)
parser.add_argument('--cosine_warm_restart_t_mult', type=float, default=1)

parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--lookahead_optimizer', type=int, default=0, help='Use Lookahead optimizer.')
parser.add_argument('--class_weight', type=str, choices=['standard', 'balanced'], default='balanced', 
                    help="If `standard`, all classes use a weight of 1.\
                            If `balanced`, classes are weighted inverse proportionally to their size (see https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)")

parser.add_argument('--debugging', action='store_true', dest='debugging')
parser.set_defaults(debugging=False)
parser.add_argument('--deterministic', action='store_true', dest='deterministic')
parser.set_defaults(deterministic=False)


####### Others
parser.add_argument('--overfit_batches', type=float, default=0, help="0 --> normal training. <1 --> overfit on % of the training data. >1 overfit on this many batches")

# SEEDS
parser.add_argument('--seed_model_init', type=int, default=42, help='Seed for initializing the model (to have the same weights)')
parser.add_argument('--seed_training', type=int, default=42, help='Seed for training (e.g., batch ordering)')

parser.add_argument('--seed_kfold', type=int, help='Seed used for doing the kfold in train/test split')
parser.add_argument('--seed_validation', type=int, help='Seed used for selecting the validation split.')

# Dataset loading
parser.add_argument('--num_workers', type=int, default=1, help="number of workers for loading dataset")
parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false', help='dont pin memory for data loaders')
parser.set_defaults(pin_memory=True)



####### Wandb logging
parser.add_argument('--group', type=str, help="Group runs in wand")
parser.add_argument('--job_type', type=str, help="Job type for wand")
parser.add_argument('--notes', type=str, help="Notes for wandb logging.")
parser.add_argument('--tags', nargs='+', type=str, default=[], help='Tags for wandb')
parser.add_argument('--suffix_wand_run_name', type=str, default="", help="Suffix for run name in wand")
parser.add_argument('--wandb_log_model', action='store_true', dest='wandb_log_model',
                    help='True for storing the model checkpoints in wandb')
parser.set_defaults(wandb_log_model=False)
parser.add_argument('--disable_wandb', action='store_true', dest='disable_wandb',
                    help='True if you dont want to crete wandb logs.')
parser.set_defaults(disable_wandb=False)

