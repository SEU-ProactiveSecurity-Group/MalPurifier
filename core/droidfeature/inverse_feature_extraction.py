import os
import time
import warnings
import random
import shutil
import tempfile
import subprocess
import traceback
import string
import re
import numpy as np
import networkx as nx
import torch
from core.droidfeature import Apk2features
from core.droidfeature import feature_gen
from tools import dex_manip, xml_manip, utils
from config import config, logging, ErrorHandler

random.seed(0)

logger = logging.getLogger('core.droidfeature.inverse_feature_extraction')
logger.addHandler(ErrorHandler)

TMP_DIR = '/tmp'

OP_INSERTION = '+'
OP_REMOVAL = '-'

MANIFEST = "AndroidManifest.xml"
REFLECTION_TEMPLATE = '''.class public Landroid/content/res/MethodReflection;
.super Ljava/lang/Object;
.source "MethodReflection.java"


# direct methods
.method public constructor <init>()V
    .locals 1

    .prologue
    .line 3
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    return-void
.end method
'''
DEFAULT_SMALI_DIR = 'android/content/res/'  # the path corresponds to the reflection class set above

INSERTION_STATIC_TEMPLATE = '''.method public static {newMethodName}()V  
    .locals {numLocals:d}

    .prologue
    const/4 v0, 0x0
    
    .local v0, "a":I
    const/4 v1, 0x1
    if-ne v0, v1, :cond_0
    
    :try_start_0
{argInitialization}
    {invokeType} {{{paramRegisters}}}, {apiClassName}->{methodName}({argumentTypes}){returnType}
    :try_end_0
    .catch Ljava/lang/Exception; {{:try_start_0 .. :try_end_0}} :catch_0

{varEndCont}

    goto :goto_0

    :catch_0
    move-exception v0

    :cond_0
    :goto_0
    return-void
.end method
'''

INSERTION_TEMPLATE = '''.method public static {newMethodName}()V  
    .locals {numLocals:d}

    .prologue
    const/4 v0, 0x0
    
    .local v0, "a":I
    const/4 v1, 0x1
    if-ne v0, v1, :cond_0
    
    const/4 v0, 0x0
    
    .local v0, "{varRandName}":{apiClassName}
    :try_start_0
{argInitialization}
    {invokeType} {{{paramRegisters}}}, {apiClassName}->{methodName}({argumentTypes}){returnType}
    :try_end_0
    .catch Ljava/lang/Exception; {{:try_start_0 .. :try_end_0}} :catch_0

    .end local v0    # "{varRandName}":{apiClassName}
{varEndCont}

    goto :goto_0

    :catch_0
    move-exception v0
    
    :cond_0
    :goto_0
    return-void
.end method
'''

ENTRY_METHOD_STATEMENT = 'public onBind(Landroid/content/Intent;)Landroid/os/IBinder;'
EMPTY_SERVICE_BODY = '''.class public L{fullClassName}
.super Landroid/app/Service;
.source "{className}.java"

# direct methods
.method public constructor <init>()V
    .locals 0

    .line 8
    invoke-direct {{p0}}, Landroid/app/Service;-><init>()V

    .line 9
    return-void
.end method


.method {entryMethodStatement}
    .locals 2
    .param p1, "intent"    # Landroid/content/Intent;

    .line 14
    new-instance v0, Ljava/lang/UnsupportedOperationException;

    const-string v1, "Not yet implemented"

    invoke-direct {{v0, v1}}, Ljava/lang/UnsupportedOperationException;-><init>(Ljava/lang/String;)V

    throw v0
.end method

'''

PROVIDER_TEMPLATE = '''.class public {ProviderCLS}
.super Landroid/content/ContentProvider;
.source "{CLSName}.java"


# direct methods
.method public constructor <init>()V
    .locals 1

    .prologue
    .line 3
    invoke-direct {{p0}}, Ljava/lang/Object;-><init>()V

    return-void
.end method
'''


class InverseDroidFeature(object):
    vocab, vocab_info, vocab_type = None, None, None

    def __init__(self, seed=0):
        random.seed(seed)
        meta_data_saving_dir = config.get('dataset', 'intermediate')
        naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
        self.feature_extractor = Apk2features(naive_data_saving_dir, meta_data_saving_dir)
        InverseDroidFeature.vocab, InverseDroidFeature.vocab_info, InverseDroidFeature.vocab_type = \
            self.feature_extractor.get_vocab()
        self.vocab = InverseDroidFeature.vocab
        self.vocab_info = InverseDroidFeature.vocab_info
        self.vocab_type = InverseDroidFeature.vocab_type

    def get_manipulation(self):
        """
        All features are insertable and the apis that have public descriptor can be hidden by java reflection.
        For efficiency and simplicity consideration, this function only returns a mask to filter out the apis that are non-refelectable.
        This means the value "1" in the mask vector corresponds to a removable feature, and "0" means otherwise.
        """
        manipulation = np.zeros((len(self.vocab),), dtype=np.float32)
        for i, v, v_info, v_type in zip(range(len(self.vocab)), self.vocab, self.vocab_info, self.vocab_type):
            if v_type in [feature_gen.ACTIVITY, feature_gen.SERVICE, feature_gen.RECEIVER, feature_gen.PROVIDER]:
                manipulation[i] = 1.
            if v_type == feature_gen.SYS_API and self.approx_check_public_method(v, v_info):
                manipulation[i] = 1.
        return manipulation

    def get_interdependent_apis(self):
        """
        For api insertion, no interdependent apis are considered. For api removal, getClass, getMethod and Invoke methods are used
        """
        interdependent_apis = ['Ljava/lang/Object;->getClass', 'Ljava/lang/Class;->getMethod',
                               'Ljava/lang/reflect/Method;->invoke']
        omega = [self.vocab.index(api) for api in interdependent_apis if api in self.vocab]
        return omega

    def get_api_flag(self):
        return [feature_gen.SYS_API == feature_type for feature_type in self.vocab_type]

    @staticmethod
    def merge_features(features_list1, features_list2):
        """
        inject features of list1 into list2
        """
        assert isinstance(features_list1, list) and isinstance(features_list2, list)
        for feature in features_list2:
            features_list1.append(feature)
        return features_list1

    @staticmethod
    def approx_check_public_method(word, word_info):
        assert isinstance(word, str) and isinstance(word_info, set)
        # see: https://docs.oracle.com/javase/specs/jvms/se10/html/jvms-2.html#jvms-2.12, we do not hide reflection-related API again
        if re.search(r'\<init\>|\<clinit\>', word) is None and \
                re.search(
                    r'Ljava\/lang\/reflect\/Method;->invoke|Ljava\/lang\/Object;->getClass|Ljava\/lang\/Class;->getMethod',
                    word) is None and \
                all(
                    [re.search(r'invoke\-virtual|invoke\-static|invoke\-interface', info) for info in word_info]):
            return True

    def inverse_map_manipulation(self, x_mod):
        """
        map the numerical manipulation to operation tuples (i.e., (feature, '+') or (feature, '-'))

        Parameters
        --------
        @param x_mod, numerical manipulations (i.e., perturbations) on node features x of a sample
        """
        assert isinstance(x_mod, (torch.Tensor, np.ndarray))
        if isinstance(x_mod, torch.Tensor) and (not x_mod.is_sparse):
            x_mod = x_mod.detach().cpu().numpy()

        indices = np.nonzero(x_mod)
        instruction = []
        features = list(map(self.vocab.__getitem__, indices[0]))
        manip_x = x_mod[indices]
        op_info = map(lambda v: OP_INSERTION if v > 0 else OP_REMOVAL, manip_x)
        instruction.append(tuple(zip(features, op_info)))

        return instruction

    @staticmethod
    def modify_wrapper(args):
        try:
            return InverseDroidFeature.modify(*args)
        except Exception as e:
            traceback.print_stack()
            traceback.print_exc()
            return e

    @staticmethod
    def modify(x_mod_instr, feature_path, app_path, save_dir=None):
        """
        model a sample

        Parameters
        --------
        @param x_mod_instr, a list of manipulations
        @param feature_path, String, feature file path
        @param app_path, String, app path
        @param save_dir, String, saving directory
        """
        features = feature_gen.read_from_disk(feature_path)
        feature_list, feature_info_list, feature_type_list = feature_gen.get_feature_list(features)
        assert os.path.isfile(app_path)

        with tempfile.TemporaryDirectory() as tmpdirname:
            dst_file = os.path.join(tmpdirname, os.path.splitext(os.path.basename(app_path))[0])
            cmd_response = subprocess.call("apktool -q d " + app_path + " -o " + dst_file, shell=True)
            if cmd_response != 0:
                logger.error("Unable to disassemble app {}".format(app_path))
                return
            methods = dex_manip.retrieve_methods(dst_file)
            component_modif = DroidCompModification(dst_file)
            permission_modif = DroidPermModification(dst_file)
            intent_modif = DroidIntentModification(dst_file)
            for instruction in x_mod_instr:
                for feature, op in instruction:
                    idx = InverseDroidFeature.vocab.index(feature)
                    feature_type = InverseDroidFeature.vocab_type[idx]
                    feature_info = InverseDroidFeature.vocab_info[idx]
                    if op == OP_REMOVAL:
                        assert feature in feature_list
                        if feature_type in [feature_gen.ACTIVITY, feature_gen.SERVICE, feature_gen.RECEIVER,
                                            feature_gen.PROVIDER]:
                            component_modif.remove(feature, feature_type)
                        elif feature_type == feature_gen.SYS_API:
                            remove_api(feature, dst_file)
                        else:
                            raise ValueError("{} is not permitted to be removed".format(feature_type))
                    else:
                        # A large scale of insertion operations will trigger unexpected issues, such as method
                        # limitation in a class
                        if feature_type in [feature_gen.ACTIVITY, feature_gen.SERVICE, feature_gen.RECEIVER]:
                            component_modif.insert(feature, feature_type)
                        elif feature_type == feature_gen.PROVIDER:
                            feature_info_list = [i for i in list(feature_info) if i]
                            assert len(feature_info_list) > 0
                            provider_cont = random.choice(feature_info_list)
                            component_modif.insert(provider_cont, feature_type)
                            # get full name and source file name
                            if feature.startswith('.'):
                                cls_full_name = xml_manip.get_package_name(os.path.join(dst_file, MANIFEST)) + feature
                                jv_src_name = feature.rstrip('.')
                            elif '.' not in feature:
                                cls_full_name = xml_manip.get_package_name(
                                    os.path.join(dst_file, MANIFEST)) + '.' + feature
                                jv_src_name = feature
                            elif '.' in feature.rstrip('.'):
                                cls_full_name = feature
                                jv_src_name = feature.rsplit('.', 1)[-1]
                            else:
                                raise ValueError("Un-support the provider '{}'".format(feature))
                            obs_path = dst_file + '/smali/' + dex_manip.name2path(cls_full_name) + '.smali'
                            os.makedirs(os.path.dirname(obs_path), exist_ok=True)
                            provider_cls = PROVIDER_TEMPLATE.format(
                                ProviderCLS=xml_manip.java_class_name2smali_name(cls_full_name),
                                CLSName=jv_src_name)
                            dex_manip.write_whole_file(provider_cls, obs_path)
                        elif feature_type == feature_gen.PERMISSION:
                            permission_modif.insert(feature, feature_gen.PERMISSION)
                        elif feature_type == feature_gen.HARDWARE:
                            permission_modif.insert(feature, feature_gen.HARDWARE)
                        elif feature_type == feature_gen.INTENT:
                            component_type = 'activity'
                            feature_info_list = [i for i in list(feature_info) if i]
                            if len(feature_info_list) > 0:
                                component_type = feature_info_list[0].split(feature_gen.TAG_SPLITTER)[0]
                            intent_modif.insert(feature, component_type)
                        elif feature_type == feature_gen.SYS_API:
                            if len(methods) <= 0:
                                warnings.warn("No space for method injection.")
                            method = random.choice(methods)[0]
                            insert_api(feature, method)

            dst_file_apk = os.path.join(save_dir, os.path.splitext(os.path.basename(app_path))[0] + '_adv')
            cmd_response = subprocess.call("apktool -q b " + dst_file + " -o " + dst_file_apk, shell=True)
            if cmd_response != 0:
                if os.path.exists(os.path.join(TMP_DIR, os.path.basename(dst_file))):
                    shutil.rmtree(os.path.join(TMP_DIR, os.path.basename(dst_file)))
                shutil.copytree(dst_file, os.path.join(TMP_DIR, os.path.basename(dst_file)))
                logger.error("Unable to assemble app {} and move it to {}.".format(dst_file, TMP_DIR))
                return False
            else:
                subprocess.call("jarsigner -sigalg MD5withRSA -digestalg SHA1 -keystore " + os.path.join(
                    config.get("DEFAULT", 'project_root'), "core/droidfeature/res/resignKey.keystore") + \
                                " -storepass resignKey " + dst_file_apk + ' resignKey',
                                shell=True)
                logger.info("Apk signed: {}.".format(dst_file_apk))
                return True


def remove_api(api_name, disassemble_dir):
    """
    remove an api

    Parameters
    --------
    @param api_name, composite of class name and method name
    @param disassemble_dir, the work directory
    @param coarse, whether use reflection to all matched methods
    """
    # we attempt to obtain more relevant info about this api. Nonetheless, once there is class inheritance,
    # we cannot make it.
    api_tag_set = set()
    api_info_list = dex_manip.retrieve_api_caller_info(api_name, disassemble_dir)
    for api_info in api_info_list:
        api_tag_set.add(feature_gen.get_api_tag(api_info['ivk_method'],
                                                api_info['caller_cls_name'],
                                                api_info['caller_mth_stm']
                                                )
                        )
    if len(api_tag_set) <= 0:
        warnings.warn("Cannot retrieve the {} in the disassemble folder {}".format(api_name, disassemble_dir))
        return
    api_class_set = set()
    for api_tag in api_tag_set:
        api_class_set.add(feature_gen.get_api_class(api_tag))

    for api_tag in api_tag_set:
        caller_class_name, caller_method_statement = feature_gen.get_caller_info(api_tag)
        smali_dirs = dex_manip.retrieve_smali_dirs(disassemble_dir)
        smali_path_of_class = None
        for smali_dir in smali_dirs:
            _path = os.path.join(smali_dir, caller_class_name.lstrip('L').rstrip(';') + '.smali')
            if os.path.exists(_path):
                smali_path_of_class = _path
                break
        if smali_path_of_class is None:
            logger.warning('File {} has no root call {}.'.format(disassemble_dir,
                                                                 caller_class_name.lstrip('L').rstrip(';') + '.smali'))
            continue
        method_finder_flag = False
        fh = dex_manip.read_file_by_fileinput(smali_path_of_class, inplace=True)
        for line in fh:
            if line.strip() == caller_method_statement:
                method_finder_flag = True
                print(line.rstrip())
                continue
            if line.strip() == '.end method':
                method_finder_flag = False

            if method_finder_flag:
                invoke_match = re.search(
                    r'^([ ]*?)(?P<invokeType>invoke\-([^ ]*?)) {(?P<invokeParam>([vp0-9,. ]*?))}, (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
                    line)
                if invoke_match is None:
                    print(line.rstrip())
                else:
                    invoked_mth_name = invoke_match.group('invokeMethod')
                    invoked_cls_name = invoke_match.group('invokeObject')
                    if (invoked_mth_name == api_name.split('->')[1]) and (invoked_cls_name in api_class_set):
                        # ivk_type + ivk_object + ivk_method + ivk_argument + ivk_return
                        cur_api_name = invoke_match.group('invokeObject') + '->' + invoke_match.group('invokeMethod')
                        new_file_name = 'Ref' + dex_manip.random_name(seed=int(time.time()),
                                                                      code=invoke_match.group(
                                                                          'invokeType') + cur_api_name + \
                                                                           invoke_match.group(
                                                                               'invokeArgument') + invoke_match.group(
                                                                          'invokeReturn'))
                        new_class_name = 'L' + DEFAULT_SMALI_DIR + new_file_name + ';'
                        ref_class_body = REFLECTION_TEMPLATE.replace('MethodReflection', new_file_name)
                        ref_class_body = dex_manip.change_invoke_by_ref(new_class_name,
                                                                        ref_class_body,  # append method
                                                                        invoke_match.group('invokeType'),
                                                                        invoke_match.group('invokeParam'),
                                                                        invoke_match.group('invokeObject'),
                                                                        invoke_match.group('invokeMethod'),
                                                                        invoke_match.group('invokeArgument'),
                                                                        invoke_match.group('invokeReturn')
                                                                        )
                        ref_smail_path = os.path.join(disassemble_dir + '/smali',
                                                      DEFAULT_SMALI_DIR + new_file_name + '.smali')
                        if not os.path.exists(os.path.dirname(ref_smail_path)):
                            utils.mkdir(os.path.dirname(ref_smail_path))
                        dex_manip.write_whole_file(ref_class_body, ref_smail_path)
                    else:
                        print(line.rstrip())
            else:
                print(line.rstrip())
        fh.close()


def create_entry_point(disassemble_dir):
    """
    creat an empty service for injecting methods
    """
    service_name = dex_manip.random_name(int(time.time())) + dex_manip.random_name(int(time.time()) + 1)
    xml_tree = xml_manip.get_xmltree_by_ET(os.path.join(disassemble_dir, MANIFEST))
    msg, response, new_manifest_tree = xml_manip.insert_comp_manifest(xml_tree, 'service', service_name)
    if not response:
        logger.error("Unable to create a new entry point {}.".format(msg))
    else:
        # create the service class correspondingly
        package_name = xml_tree.getroot().get('package')
        full_classname = package_name.replace('.', '/') + service_name + ';'
        service_class_body = EMPTY_SERVICE_BODY.format(
            fullClassName=full_classname,
            className=service_name,
            entryMethodStatement=ENTRY_METHOD_STATEMENT
        )

        svc_class_path = os.path.join(disassemble_dir + '/smali',
                                      package_name.replace('.', '/') + '/' + service_name + '.smali')
        if not os.path.exists(svc_class_path):
            dex_manip.mkdir(os.path.dirname(svc_class_path))
        dex_manip.write_whole_file(service_class_body, svc_class_path)
        return 'L' + full_classname + '.method ' + ENTRY_METHOD_STATEMENT


def insert_api(api_name, method_location):
    """
    insert an api.

    Parameters
    -------
    @param api_name, composite of class name and method name
    @param method_location, work file
    """
    api_info = InverseDroidFeature.vocab_info[InverseDroidFeature.vocab.index(api_name)]
    class_name, method_name = api_name.split('->')

    invoke_types, return_classes, arguments = list(), list(), list()
    for info in list(api_info):
        _match = re.search(
            r'^([ ]*?)(?P<invokeType>invoke\-([^ ]*?)) (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
            info)
        invoke_types.append(_match.group('invokeType'))
        return_classes.append(_match.group('invokeReturn'))
        arguments.append(_match.group('invokeArgument'))

    api_idx = 0
    is_simplified_vars_register = False
    if 'invoke-virtual' in invoke_types:
        invoke_type = 'invoke-virtual'
        api_idx = invoke_types.index('invoke-virtual')
    elif 'invoke-virtual/range' in invoke_types:
        invoke_type = 'invoke-virtual/range'
        api_idx = invoke_types.index('invoke-virtual/range')
        is_simplified_vars_register = True
    elif 'invoke-interface' in invoke_types:
        invoke_type = 'invoke-interface'
        api_idx = invoke_types.index('invoke-interface')
    elif 'invoke-interface/range' in invoke_types:
        invoke_type = 'invoke-interface/range'
        api_idx = invoke_types.index('invoke-interface/range')
        is_simplified_vars_register = True
    elif 'invoke-static' in invoke_types:
        invoke_type = 'invoke-static'
    elif 'invoke-static/range' in invoke_types:
        invoke_type = 'invoke-static/range'
        is_simplified_vars_register = True
    elif 'invoke-super' in invoke_types:
        invoke_type = 'invoke-super'
        api_idx = invoke_types.index('invoke-super')
    elif 'invoke-super/range' in invoke_types:
        invoke_type = 'invoke-super/range'
        api_idx = invoke_types.index('invoke-super/range')
        is_simplified_vars_register = True
    elif 'invoke-direct' in invoke_types:
        invoke_type = 'invoke-direct'
        api_idx = invoke_types.index('invoke-direct')
    elif 'invoke-direct/range' in invoke_types:
        invoke_type = 'invoke-direct/range'
        api_idx = invoke_types.index('invoke-direct/range')
        is_simplified_vars_register = True
    else:
        logger.warning('Neglect invocation type(s):{}'.format(' '.join(invoke_types)))
        return

    if method_name == '<init>':
        logger.warning('Unable to add <init> method:{}'.format(api_name))
        return

    assert len(invoke_types) > 0, 'No api details.'
    return_class = return_classes[api_idx]
    argument = arguments[api_idx]
    # handle arguments
    # variable initialization
    arg_types = argument.split(' ')  # this is specific to androguard
    var_initial_content = ''
    if 'invoke-static' not in invoke_type:
        var_registers = 'v0'
    else:
        var_registers = ''
    var_count = 0
    var_end_content = ''
    for arg_type in arg_types:
        arg_type = arg_type.strip()
        if arg_type == '':
            continue
        if var_count >= 52:
            raise ValueError("Too much arguments of API method {}.".format(api_name))
        if arg_type[-1] == ';':
            var_count += 1
            var_value = dex_manip.smaliClassTInitialV.format(varNum=var_count)
            var_statement = dex_manip.VAR_STATEMENT_TEMPLATE.format(varNum=var_count,
                                                                    varName=string.ascii_letters[var_count - 1],
                                                                    varType=arg_type)
            var_end = dex_manip.VAR_END_TEMPLATE.format(varNum=var_count,
                                                        varName=string.ascii_letters[var_count - 1],
                                                        varType=arg_type
                                                        )
            var_registers += ', v{:d}'.format(var_count)
        elif arg_type in list(dex_manip.smaliBasicTInitialV.keys()):
            var_count += 1
            var_value = dex_manip.smaliBasicTInitialV[arg_type].format(varNum=var_count)
            var_statement = dex_manip.VAR_STATEMENT_TEMPLATE.format(varNum=var_count,
                                                                    varName=string.ascii_letters[var_count - 1],
                                                                    varType=arg_type)
            var_end = dex_manip.VAR_END_TEMPLATE.format(varNum=var_count,
                                                        varName=string.ascii_letters[var_count - 1],
                                                        varType=arg_type
                                                        )
            var_registers += ', v{:d}'.format(var_count)
            if dex_manip.is_wide_type(arg_type):  # 'const-wide'
                var_count += 1
                var_registers += ', v{:d}'.format(var_count)

        elif arg_type in list(dex_manip.smaliArrayInitialV.keys()):
            var_count += 1
            var_value = dex_manip.smaliArrayInitialV[arg_type].format(varNum=var_count)
            var_statement = dex_manip.VAR_STATEMENT_TEMPLATE.format(varNum=var_count,
                                                                    varName=string.ascii_letters[var_count - 1],
                                                                    varType=arg_type)
            var_end = dex_manip.VAR_END_TEMPLATE.format(varNum=var_count,
                                                        varName=string.ascii_letters[var_count - 1],
                                                        varType=arg_type
                                                        )
            var_registers += ', v{:d}'.format(var_count)

        var_initial_content += '\n' + var_value + '\n' + var_statement + '\n'
        var_end_content += var_end + '\n'
    var_registers = var_registers.lstrip(',').strip()
    arg_types_used = ''.join(arg_types[:var_count])

    random_str = dex_manip.random_name(seed=int(time.time()), code=api_name)
    # handle the initialization methods: <init>, <cinit>
    new_method_name = method_name.lstrip('<').rstrip('>') + random_str
    if invoke_type != 'invoke-static' and invoke_type != 'invoke-static/range':
        if var_count >= 5 and '/range' not in invoke_type:
            invoke_type += '/range'
            is_simplified_vars_register = True
        if is_simplified_vars_register:
            var_registers = 'v0 .. v{:d}'.format(var_count)
        var_count = var_count + 1 if var_count >= 1 else 2
        new_method_body = INSERTION_TEMPLATE.format(
            newMethodName=new_method_name,
            numLocals=var_count,
            argInitialization=var_initial_content,
            methodName=method_name,
            varRandName=random_str,
            invokeType=invoke_type,
            paramRegisters=var_registers,
            apiClassName=class_name,
            argumentTypes=arg_types_used,
            returnType=return_class,
            varEndCont=var_end_content
        )
    else:
        if var_count > 5 and '/range' not in invoke_type:
            invoke_type += '/range'
            is_simplified_vars_register = True
        if is_simplified_vars_register:
            var_registers = 'v1 .. v{:d}'.format(var_count)
        var_count = var_count + 1 if var_count >= 1 else 2
        new_method_body = INSERTION_STATIC_TEMPLATE.format(
            newMethodName=new_method_name,
            numLocals=var_count,
            argInitialization=var_initial_content,
            methodName=method_name,
            varRandName=random_str,
            invokeType=invoke_type,
            paramRegisters=var_registers,
            apiClassName=class_name,
            argumentTypes=arg_types_used,
            returnType=return_class,
            varEndCont=var_end_content
        )

    smali_path, class_name, a_method_statement = method_location
    if smali_path is None:
        logger.warning('smali file {} is absent.'.format(smali_path))

    method_finder_flag = False
    fh = dex_manip.read_file_by_fileinput(smali_path, inplace=True)
    for line in fh:
        print(line.rstrip())

        if line.strip() == a_method_statement:
            method_finder_flag = True
            continue

        if method_finder_flag and line.strip() == '.end method':
            method_finder_flag = False
            # issue: injection ruins the correct line number in smali codes
            print('\n')
            print(new_method_body)
            continue

        invoke_static = 'invoke-static'
        if method_finder_flag and '.locals' in line:
            reg_match = re.match(r'^[ ]*?(.locals)[ ]*?(?P<regNumber>\d+)', line)
            if reg_match is not None and int(reg_match.group('regNumber')) > 15:
                invoke_static = 'invoke-static/range'

        if method_finder_flag:
            if re.match(r'^[ ]*?(.locals)', line) is not None:
                print(
                    '    ' + invoke_static + ' {}, ' + class_name + '->' + new_method_name + '()V' + '\n')
    fh.close()


class DroidCompModification:
    """Modification for components"""

    def __init__(self, disassembly_root):
        self.disassembly_root = disassembly_root

    def insert(self, specific_name, component_type):
        """
        Insert an component based on 'specfic_name' into manifest.xml file
        """
        if not isinstance(specific_name, str) and not os.path.isdir(self.disassembly_root):
            raise TypeError("Type error: expect string but got {}.".format(type(specific_name)))

        if '..' in specific_name:
            specific_name = specific_name.split('..')[-1]
            specific_name = '.' + specific_name
        else:
            specific_name = specific_name

        manifest_tree = xml_manip.get_xmltree_by_ET(os.path.join(self.disassembly_root, MANIFEST))

        if component_type != feature_gen.PROVIDER:
            info, flag, new_manifest_tree = xml_manip.insert_comp_manifest(manifest_tree,
                                                                           component_type,
                                                                           specific_name,
                                                                           mod_count=1)
        else:
            info, flag, new_manifest_tree = xml_manip.insert_provider_manifest(manifest_tree,
                                                                               specific_name,
                                                                               mod_count=1)

        xml_manip.dump_xml(os.path.join(self.disassembly_root, MANIFEST), new_manifest_tree)
        if flag:
            logger.info(
                "Component insertion: Successfully insert {} '{}' into '{}'/androidmanifest.xml".format(
                    component_type,
                    specific_name,
                    os.path.basename(self.disassembly_root)))
        else:
            logger.warning(info)

    def _rename_files(self, smali_paths, activity_name, new_activity_name):
        for smali_path in smali_paths:
            orginal_name = activity_name.split('.')[-1]
            modified_name = new_activity_name.split('.')[-1]
            if orginal_name in smali_path and \
                    modified_name not in smali_path:
                dex_manip.rename_smali_file(smali_path,
                                            activity_name,
                                            new_activity_name)

    def _rename_folders(self, smali_dirs, activity_name, new_activity_name):
        for smali_dir in smali_dirs:
            if dex_manip.name2path(activity_name) in smali_dir or \
                    os.path.dirname(dex_manip.name2path(activity_name)) in smali_dir:
                dex_manip.rename_smali_dir(smali_dir,
                                           activity_name,
                                           new_activity_name)

    def remove(self, specific_name, component_type):
        """
        change the denoted component name to a random string
        mod_count = -1 indicates that all the corresponding elements will be changed
        """

        # step 1: modify the corresponding name in AndroidManifest.xml
        if not isinstance(specific_name, str) and not os.path.exists(self.disassembly_root):
            raise ValueError("Value error:")

        # mod_count = -1 change name of all the specified components
        manifest_tree = xml_manip.get_xmltree_by_ET(os.path.join(self.disassembly_root, MANIFEST))
        if '..' in specific_name:
            specific_name = specific_name.split('..')[-1]
            specific_name = '.' + specific_name
        elif '.' in specific_name:
            name_rear = specific_name.rsplit('.', 1)[-1]
            if xml_manip.check_comp_name(manifest_tree, component_type, specific_name):
                specific_name = specific_name
            elif xml_manip.check_comp_name(manifest_tree, component_type, name_rear):
                specific_name = name_rear
            else:
                pass
        else:
            pass

        info, flag, new_comp_name, new_manifest_tree = xml_manip.rename_comp_manifest(manifest_tree,
                                                                                      component_type,
                                                                                      specific_name)
        xml_manip.dump_xml(os.path.join(self.disassembly_root, MANIFEST), new_manifest_tree)

        if flag:
            logger.info(
                "'{}' name changing: Successfully change name '{}' to '{}' of '{}'/androidmanifest.xml".format(
                    component_type,
                    specific_name,
                    new_comp_name,
                    os.path.basename(self.disassembly_root)
                ))
        else:
            logger.warning(info + ": {}/androidmanifest.xml".format(os.path.basename(self.disassembly_root)))

        # step 2: modify .smali files accordingly
        package_name = manifest_tree.getroot().get('package')
        smali_paths = dex_manip.get_smali_paths(self.disassembly_root)
        related_smali_paths = set(dex_manip.find_smali_w_name(smali_paths, specific_name))
        dex_manip.change_source_name(related_smali_paths, specific_name, new_comp_name)
        changed_class_names = set(dex_manip.change_class_name(related_smali_paths,
                                                              specific_name,
                                                              new_comp_name,
                                                              package_name))

        # Change class instantiation
        if len(changed_class_names) > 0:
            dex_manip.change_instantition_name(smali_paths,
                                               changed_class_names,
                                               specific_name,
                                               new_comp_name,
                                               package_name)

        # step 3: modify all .xml files accordingly
        # if len(changed_class_names) > 0:
        #     xml_paths = xml_manip.get_xml_paths(self.disassembly_root)
        #     xml_manip.change_xml(xml_paths, changed_class_names,
        #                          specific_name, new_comp_name, package_name)

        # step 3: modify folder and file names
        self._rename_files(smali_paths, specific_name, new_comp_name)

        logger.info("'{}' name changing: Successfully done '{}'".format(
            component_type,
            specific_name))


class DroidPermModification(object):
    """Modification for permission and hardware"""

    def __init__(self, disassembly_root):
        self.disassembly_root = disassembly_root

    def insert(self, specific_name, feature_type):
        '''Insert a permission of 'specfic_name' into manifest.xml file'''
        if feature_type == feature_gen.PERMISSION:
            feature_type = 'uses-permission'
        elif feature_type == feature_gen.HARDWARE:
            feature_type = 'uses-feature'
        else:
            raise ValueError("Expected '{}' \& '{}', but got '{}'".format(feature_gen.PERMISSION,
                                                                          feature_gen.HARDWARE,
                                                                          feature_type))

        manifest_tree = xml_manip.get_xmltree_by_ET(os.path.join(self.disassembly_root, MANIFEST))

        info, flag, new_manifest_tree = xml_manip.insert_perm_manifest(manifest_tree,
                                                                       feature_type,
                                                                       specific_name,
                                                                       mod_count=1)

        xml_manip.dump_xml(os.path.join(self.disassembly_root, MANIFEST), new_manifest_tree)

        if flag:
            logger.info(
                'Permission insertion: Successfully insert \'{}\' into \'{}/androidmanifest.xml\''.format(
                    specific_name,
                    os.path.basename(self.disassembly_root)))
        else:
            logger.warning(info)

    def remove(self, elem_name):
        raise NotImplementedError("Risk the functionality.")


class DroidIntentModification(object):
    def __init__(self, disassembly_root):
        self.disassembly_root = disassembly_root

    def insert(self, specific_name, component_type='activity'):
        """
        Insert an intent-filter of 'specfic_name' into AndroidManifest.xml file
        """
        if not isinstance(specific_name, str) and not os.path.isdir(self.disassembly_root):
            raise ValueError("Value error: require str type of variables.")

        manifest_tree = xml_manip.get_xmltree_by_ET(os.path.join(self.disassembly_root, MANIFEST))

        info, flag, new_manifest_tree = xml_manip.insert_intent_manifest(manifest_tree,
                                                                         component_type,
                                                                         specific_name,
                                                                         mod_count=1)
        xml_manip.dump_xml(os.path.join(self.disassembly_root, MANIFEST), new_manifest_tree)

        if flag:
            logger.info(
                "intent-filter insertion: Successfully insert intent-filter '{}' into '{}'/androidmanifest.xml".format(
                    specific_name,
                    os.path.basename(self.disassembly_root)))
        else:
            logger.error(info)

    def remove(self, elem_name, mod_count=1):
        raise NotImplementedError
