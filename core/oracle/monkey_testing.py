import os
import sys
import warnings

import shutil
import subprocess
import time
from collections import defaultdict
from tools import utils

from config import logging, ErrorHandler

logger = logging.getLogger('core.oracle.fuzz_testing_monkey')
logger.addHandler(ErrorHandler)


class APKTestADB(object):
    def __init__(self, cache_dir="/tmp/apk_test_adb/"):
        """
        Get results from adb testing, including app installablity, app runnability, and behaviours comparison between two apps
        :param cache_dir: the default directory of saving log files
        """
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            utils.mkdir(self.cache_dir)

        self.temp_apk_dir = os.path.join(self.cache_dir, 'apk_dir')
        if not os.path.exists(self.temp_apk_dir):
            utils.mkdir(self.temp_apk_dir)

        self.temp_res_dir = os.path.join(self.cache_dir, 'res_dir')
        if not os.path.exists(self.temp_res_dir):
            utils.mkdir(self.temp_res_dir)

        self._check_device_state()

    def _check_device_state(self):
        # get device/emulator state
        try:
            proc_res = subprocess.check_output(['adb', 'get-state']).decode('utf-8')
            if proc_res.strip() == "device":  # support a device
                logger.info("Ready for testing...")
                self.state = True
        except subprocess.CalledProcessError as e:
            logger.warning("No suitable devices {}. Exit!".format(e.output))
            self.state = False
            sys.exit(1)

    def _check_file_existence(self, apk_path):
        if not os.path.exists(apk_path):
            raise IOError("File: {} not found.".format(apk_path))
        if not os.path.splitext(apk_path)[-1] == '.apk':
            raise IOError("'.apk' format needed")

    def _get_pkg_name(self, apk_path):
        try:
            self._check_file_existence(apk_path)
            full_pkg_name = subprocess.check_output("aapt dump badging " + apk_path + " | grep package:\ name",
                                                    shell=True, stderr=subprocess.STDOUT).decode('utf-8')
            if full_pkg_name.startswith('package'):
                full_pkg_name = full_pkg_name.lstrip('package ')
                pkg_name = \
                    [info.lstrip('name=').strip("'") for info in full_pkg_name.split(' ') if info.startswith('name')][0]
                return pkg_name
            elif 'package: name' in full_pkg_name:
                pkg_name = \
                    [info.lstrip('name=').strip("'") for info in full_pkg_name.split(' ') if info.startswith('name')][0]
                return pkg_name
            else:
                logger.error("No package name found: " + apk_path)
                return 'no-pkg-name-return'
        except subprocess.CalledProcessError as e:
            logger.error(str(e) + ": " + apk_path)
            return 'no-pkg-name-return'

    def _get_saving_path(self, apk_path):
        return os.path.join(self.temp_res_dir,
                            self._get_pkg_name(apk_path) + utils.get_sha256(apk_path) + '.json')

    def _check_apk_on_device(self, pkg_name):
        proc_out = subprocess.check_output('adb shell pm list packages', shell=True,
                                           stderr=subprocess.STDOUT).decode('utf-8')

        if (proc_out is not None) and (pkg_name in 'package:' + pkg_name in proc_out.strip()):
            return True
        else:
            return False

    def install_apk(self, apk_path):
        self._check_file_existence(apk_path)

        sha256 = utils.get_sha256(apk_path)  # identification
        pkg_name = self._get_pkg_name(apk_path)
        i_log_dir = os.path.join(self.cache_dir, 'install_logs')
        if not os.path.exists(i_log_dir):
            utils.mkdir(i_log_dir)
        if self._check_apk_on_device(pkg_name):
            warnings.warn("An apk with same package name has existed on the device. Uninstallation performed.")
            self.remove_apk(apk_path)

        os.system('adb logcat -c -b main -b events -b radio')
        try:
            proc_res = subprocess.check_output(['adb', 'install', apk_path],
                                               stderr=subprocess.STDOUT).decode('utf-8')  # return "Success\n"
            if proc_res.strip().lower() == 'success':
                logger.info("Succeed to install the app: {}".format(os.path.basename(apk_path)))
                return True
            else:
                log_fpath = os.path.join(i_log_dir, sha256 + ".install")
                os.system('adb' + ' logcat -d >>' + log_fpath)
                logger.info("Fail to install the app: {}".format(os.path.basename(apk_path)))
                return False
        except subprocess.CalledProcessError as e:
            log_fpath = os.path.join(i_log_dir, sha256 + ".install")
            os.system('adb' + ' logcat -d >>' + log_fpath)
            logger.error("Command '{}' return with error (code {}):{}".format(e.cmd, e.returncode, e.output))
            return False

    def remove_apk(self, apk_path):
        self._check_file_existence(apk_path)

        sha256 = utils.get_sha256(apk_path)  # idenfication
        r_log_dir = os.path.join(self.cache_dir, 'remove_logs')
        if not os.path.exists(r_log_dir):
            utils.mkdir(r_log_dir)
        os.system('adb logcat -c -b main -b events -b radio')

        try:
            pkg_name = self._get_pkg_name(apk_path)
            # stop the running apk
            subprocess.check_output(['adb', 'shell', 'am', 'force-stop', pkg_name]).decode('utf-8')
            # uninstall the apk
            proc_res = subprocess.check_output(['adb', 'uninstall', pkg_name], stderr=subprocess.STDOUT).decode('utf-8')

            if proc_res.strip().lower() == 'success':
                logger.info("Succeed to uninstall the app: {}".format(os.path.basename(apk_path)))
                return True
            else:
                log_fpath = os.path.join(r_log_dir, sha256 + ".uninstall")
                os.system('adb' + ' logcat -d >>' + log_fpath)
                os.system('adb' + ' logcat -c')
                logger.info("Fail to uninstall the app: {}".format(os.path.basename(apk_path)))
                return False
        except subprocess.CalledProcessError as e:
            log_fpath = os.path.join(r_log_dir, sha256 + ".uninstall")
            os.system('adb' + ' logcat -d >>' + log_fpath)
            os.system('adb' + ' logcat -c')
            logger.error("Command '{}' return with error (code {}):{}".format(e.cmd, e.returncode, e.output))
            return False

    def run_monkey(self, apk_path, count=1000, seed=123456543):
        self._check_file_existence(apk_path)
        pkg_name = self._get_pkg_name(apk_path)
        sha256 = utils.get_sha256(apk_path)

        # check apk existence
        if not self._check_apk_on_device(pkg_name):
            raise IOError("No apk ({}) on the device.".format(apk_path))

        # run monkey testing
        m_log_dir = os.path.join(self.cache_dir, 'monkey_logs')
        if not os.path.exists(m_log_dir):
            utils.mkdir(m_log_dir)
        try:
            os.system('adb logcat -c -b main -b events -b radio')
            os.system('adb logcat -c')
            try:
                proc_out = subprocess.check_output(
                    ['adb', 'shell', 'monkey', '-p', pkg_name, '--ignore-crashes', '--ignore-timeouts',
                     '--ignore-security-exceptions', '--pct-appswitch', '80', '-s', str(seed), '-v', '-v',
                     '-v', str(count)]).decode('utf-8')
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "Command '{}' return with error (code {}):{}".format(e.cmd, e.returncode, e.output))
            logger.info('Handling the apk {}.'.format(os.path.basename(apk_path)))
            logger.info('Done: {}'.format(os.path.basename(apk_path)))
            proc_adb_log = subprocess.run('adb logcat -d',
                                          shell=True,
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT,
                                          timeout=500,
                                          ).stdout.decode('utf-8')
        except subprocess.CalledProcessError as e:
            log_fpath = os.path.join(m_log_dir, sha256 + ".monkey")
            os.system('adb' + ' logcat -d >>' + log_fpath)
            raise RuntimeError("Command '{}' return with error (code {}):{}".format(e.cmd, e.returncode, e.output))

        proc_res_activites = []
        proc_res_exps = []
        is_next_line_exception = False
        for line in proc_adb_log.split('\n'):
            if ('I ActivityManager: Displayed' in line) and (pkg_name in line):
                proc_res_activites.append(line)
            elif ('AndroidRuntime' in line) and ('Process' in line) and (pkg_name in line):
                is_next_line_exception = True
            elif is_next_line_exception:
                proc_res_exps.append(line)
                is_next_line_exception = False
            else:
                continue

        activities = []
        exceptions = []

        for line in proc_res_activites:
            activities.append(line.split('I ActivityManager: Displayed')[1].strip().split(":")[0])

        for line in proc_res_exps:
            exceptions.append(line.split('AndroidRuntime: ')[1].strip().split(':')[0])
        return activities, exceptions

    def submit(self, apk_path):
        """
        submit an apk for estimation
        :param apk_path: apk path of local disk
        :return: state
        """
        shutil.copy(apk_path, os.path.join(self.temp_apk_dir,
                                           self._get_pkg_name(apk_path) + utils.get_sha256(apk_path) + '.apk'))
        time.sleep(5)
        return

    def run(self):
        """
        Analyze APK, exit by [Ctrl+C]
        :return: None
        """
        while True:
            try:
                apk_names = os.listdir(self.temp_apk_dir)
                for i, n in enumerate(apk_names):
                    apk_path = os.path.join(self.temp_apk_dir, n)
                    save_path = self._get_saving_path(apk_path)
                    if os.path.exists(save_path):
                        os.remove(apk_path)
                        continue
                    info = defaultdict()
                    info['install'] = ''
                    info['components'] = ''
                    info['exceptions'] = ''
                    res = self.install_apk(apk_path)
                    if res:
                        info['install'] = 'success'
                        try:
                            cmps, exps = self.run_monkey(apk_path)
                            self.remove_apk(apk_path)
                            info['components'] = ','.join(cmps)
                            info['exceptions'] = ','.join(exps)
                            utils.dump_json(info, save_path)

                            os.remove(apk_path)
                        except Exception as e:
                            utils.dump_json(info, save_path)
                            logger.error(str(e))
                    else:
                        info['install'] = 'failure'
                        utils.dump_json(info, save_path)
                        logger.error("Cannot install application {}.".format(os.path.basename(apk_path)))
                        # sys.exit(1)
            except Exception as e:
                logger.error(str(e))
            print("No queries.")
            time.sleep(5)

    def get_state(self, apk_path):
        """
        The specific analysis is ready or not
        :param apk_path: the specific apk
        :return: Ture or False
        """
        save_path = self._get_saving_path(apk_path)
        if os.path.exists(save_path):
            return True
        else:
            return False

    def get_report(self, apk_path):
        """
        return the set of components and exceptions
        :param apk_path: apk path
        :return: set of component, set of exceptions
        :rtype: set,set 
        """
        save_path = self._get_saving_path(apk_path)
        if self.get_state(apk_path):
            json_res = utils.load_json(save_path)
            install_flag = json_res['install'] == 'success'
            component_set = set(json_res['components'].split(','))
            exception_set = set(json_res['exceptions'].split(','))
            return install_flag, component_set, exception_set
        else:
            warnings.warn("Result is not ready.")
            return set([]), set([])


def _main():
    apk_test_adb = APKTestADB()
    # apk_test_adb.install_apk("/local_disk/data/Android//drebin/malicious_samples/2dd94e49e75467cb055099319fc90d0f93d0868e531593f857cf91f6f3220c87.apk")
    # apk_test_adb.run_monkey("/local_disk/data/Android//drebin/malicious_samples/2dd94e49e75467cb055099319fc90d0f93d0868e531593f857cf91f6f3220c87.apk")
    # apk_test_adb.remove_apk("/local_disk/data/Android//drebin/malicious_samples/2dd94e49e75467cb055099319fc90d0f93d0868e531593f857cf91f6f3220c87.apk")
    # apk_test_adb.submit('/local_disk/data/Android//drebin/malicious_samples/2dd94e49e75467cb055099319fc90d0f93d0868e531593f857cf91f6f3220c87.apk')
    apk_test_adb.run()
    apk_test_adb.get_state(
        "/local_disk/data/Android//drebin/malicious_samples/2dd94e49e75467cb055099319fc90d0f93d0868e531593f857cf91f6f3220c87.apk")
    cmps, exps = apk_test_adb.get_report(
        "/local_disk/data/Android//drebin/malicious_samples/2dd94e49e75467cb055099319fc90d0f93d0868e531593f857cf91f6f3220c87.apk")
    print(cmps, exps)


if __name__ == "__main__":
    _main()
