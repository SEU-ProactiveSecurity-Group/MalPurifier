import os
import numpy as np
import tempfile
import shutil
import time
import argparse
from core.oracle import APKTestADB

from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.monkey_testing')
logger.addHandler(ErrorHandler)

# Parse command line arguments for monkey testing
monkeyt_argparse = argparse.ArgumentParser(description='Arguments for monkey testing')
monkeyt_argparse.add_argument('--n_samples', type=int, default=100,
                              help='Number of samples for functionality testing.')
monkeyt_argparse.add_argument('--attacks', type=str, default='max,mimicry,stepwise_max',
                              help="Attack types: max, mimicry, stepwise_max, etc.")
monkeyt_argparse.add_argument('--models', type=str, default='amd_pad_ma',
                              choices=['md_dnn', 'md_at_pgd', 'md_at_ma',
                                       'amd_kde', 'amd_icnn', 'amd_dla', 'amd_dnn_plus', 'amd_pad_ma'],
                              help="Model type to use for testing.")

def _main():
    np.random.seed(0)
    args = monkeyt_argparse.parse_args()
    attack_names = args.attacks.split(',')
    model_names = args.models.split(',')

    sample_save_dirs = []
    malicious_sample_dir = config.get('dataset', 'malware_dir')

    # Generate paths for saving adversarial samples
    for attack in attack_names:
        for model in model_names:
            save_dir = os.path.join(
                os.path.join(config.get('experiments', attack), model), 'adv_apps')
            sample_save_dirs.append(save_dir)

    # Find common apps across all attack/model combinations
    list_of_sets = [set(os.listdir(save_dir)) for save_dir in sample_save_dirs]
    inter_apps = list(set.intersection(*list_of_sets))
    n_samples = min(args.n_samples, len(inter_apps))
    app_names = np.random.choice(inter_apps, n_samples, replace=False).tolist()

    apk_test_adb = APKTestADB()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Submit original and adversarial apps for testing
        for app_name in app_names:
            app_name_ = app_name.split('_')[0]
            apk_path = os.path.join(malicious_sample_dir, app_name_ + '.apk')
            tmp_path = os.path.join(tmpdir, app_name_ + '.apk')
            shutil.copy(apk_path, tmp_path)
            logger.info("Submitting: " + apk_path)
            apk_test_adb.submit(tmp_path)
            for save_dir in sample_save_dirs:
                apk_path = os.path.join(save_dir, app_name)
                tmp_path = os.path.join(tmpdir, app_name)
                shutil.copy(apk_path, tmp_path)
                logger.info("Submitting: " + apk_path)
                apk_test_adb.submit(tmp_path)

        # Wait for all tests to complete
        while not all(apk_test_adb.get_state(os.path.join(tmpdir, app_name.split('_')[0] + '.apk')) for app_name in app_names):
            time.sleep(1)

        # Process test results
        org_sample_installed = 0
        org_sample_function = 0
        install_flag_2dlist = []
        functionality_flag_2dlist = []
        for app_name in app_names:
            app_name_ = app_name.split('_')[0]
            tmp_path = os.path.join(tmpdir, app_name_ + '.apk')
            org_install_flag, org_activities, org_exceptions = apk_test_adb.get_report(tmp_path)
            if not org_install_flag:
                logger.info(f"Unperturbed example {app_name_}: failed to install.")
                continue
            org_sample_installed += 1
            org_func_flag = not (len(org_activities) <= 1 and '' in org_activities and 
                                 len(org_exceptions) <= 1 and '' in org_exceptions)
            if org_func_flag:
                org_sample_function += 1
            else:
                logger.info(f"Unperturbed example {app_name_}: No activities and exceptions.")
                print("False:", org_activities, org_exceptions)
            
            install_flag_list = []
            func_flag_list = []
            for save_dir in sample_save_dirs:
                apk_path = os.path.join(save_dir, app_name)
                tmp_path = os.path.join(tmpdir, app_name)
                shutil.copy(apk_path, tmp_path)
                adv_install_flag, adv_activities, adv_exceptions = apk_test_adb.get_report(tmp_path)
                install_flag_list.append(adv_install_flag)
                func_flag = org_activities == adv_activities if org_func_flag else False
                if not func_flag:
                    logger.info(f"Functionality ruined: {apk_path}")
                    logger.info(f'\t Original activities: {",".join(list(org_activities))}')
                    logger.info(f'\t Perturbed activities: {",".join(list(adv_activities))}')
                func_flag_list.append(func_flag)

            install_flag_2dlist.append(install_flag_list)
            functionality_flag_2dlist.append(func_flag_list)

        # Summarize results
        install_count = np.sum(np.array(install_flag_2dlist), axis=0).tolist()
        func_count = np.sum(np.array(functionality_flag_2dlist), axis=0).tolist()
        logger.info(f"Installable apps: {org_sample_installed}; runnable apps: {org_sample_function}.")
        for i, attack in enumerate(attack_names):
            logger.info(f"Attack {attack}: number of installable apks {install_count[i]} and runnable apks {func_count[i]}.")

    return


if __name__ == "__main__":
    _main()
