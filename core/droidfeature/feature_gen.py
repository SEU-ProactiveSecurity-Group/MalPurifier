"""
Extract features for APKs
"""

from os import path, getcwd
import io
import copy
import warnings

import collections
import lxml.etree as etree
from xml.dom import minidom
from androguard.misc import AnalyzeAPK, DalvikVMFormat

import re

from collections import OrderedDict

from tools.utils import dump_pickle, read_pickle, java_class_name2smali_name, \
    read_txt, retrive_files_set, remove_duplicate

PERMISSION = 'permission'
INTENT = 'intent'
ACTIVITY = 'activity'
SERVICE = 'service'
RECEIVER = 'receiver'
PROVIDER = 'provider'
HARDWARE = 'hardware'
SYS_API = 'api'

DANGEROUS_PERMISSION_TAGS = [
    'android.permission.WRITE_CONTACTS',
    'android.permission.GET_ACCOUNTS',
    'android.permission.READ_CONTACTS',
    'android.permission.READ_CALL_LOG',
    'android.permission.READ_PHONE_STATE',
    'android.permission.CALL_PHONE',
    'android.permission.WRITE_CALL_LOG',
    'android.permission.USE_SIP',
    'android.permission.PROCESS_OUTGOING_CALLS',
    'com.android.voicemail.permission.ADD_VOICEMAIL',
    'android.permission.READ_CALENDAR',
    'android.permission.WRITE_CALENDAR',
    'android.permission.CAMERA',
    'android.permission.BODY_SENSORS',
    'android.permission.ACCESS_FINE_LOCATION',
    'android.permission.ACCESS_COARSE_LOCATION',
    'android.permission.READ_EXTERNAL_STORAGE',
    'android.permission.WRITE_EXTERNAL_STORAGE',
    'android.permission.RECORD_AUDIO',
    'android.permission.READ_SMS',
    'android.permission.RECEIVE_WAP_PUSH',
    'android.permission.RECEIVE_MMS',
    'android.permission.RECEIVE_SMS',
    'android.permission.SEND_SMS',
    'android.permission.READ_CELL_BROADCASTS'
]

INTENT_TAGS = [
    'android.intent.action',
    'com.android.vending',
    'android.net',
    'com.android'
]

DANGEROUS_API_SIMLI_TAGS = [
    'Landroid/content/Intent;->setDataAndType',
    'Landroid/content/Intent;->setFlags',
    'Landroid/content/Intent;->addFlags',
    'Landroid/content/Intent;->putExtra',
    'Landroid/content/Intent;->init',
    'Ljava/lang/reflect',
    'Ljava/lang/Object;->getClass',
    'Ljava/lang/Class;->getConstructor',
    'Ljava/lang/Class;->getConstructors',
    'Ljava/lang/Class;->getDeclaredConstructor',
    'Ljava/lang/Class;->getDeclaredConstructors',
    'Ljava/lang/Class;->getField',
    'Ljava/lang/Class;->getFields',
    'Ljava/lang/Class;->getDeclaredField',
    'Ljava/lang/Class;->getDeclaredFields',
    'Ljava/lang/Class;->getMethod',
    'Ljava/lang/Class;->getMethods',
    'Ljava/lang/Class;->getDeclaredMethod',
    'Ljava/lang/Class;->getDeclaredMethods',
    'Ljavax/crypto',
    'Ljava/security/spec',
    'Ldalvik/system/DexClassLoader',
    'Ljava/lang/System;->loadLibrary',
    'Ljava/lang/Runtime',
    'Landroid/os/Environment;->getExternalStorageDirectory',
    'Landroid/telephony/TelephonyManager;->getDeviceId',
    'Landroid/telephony/TelephonyManager;->getSubscriberId',
    'setWifiEnabled',
    'execHttpRequest',
    'getPackageInfo',
    'Landroid/content/Context;->getSystemService',
    'setWifiDisabled',
    'Ljava/net/HttpURLconnection;->setRequestMethod',
    'Landroid/telephony/SmsMessage;->getMessageBody',
    'Ljava/io/IOException;->printStackTrace',
    'system/bin/su'  # non-alike an api but emerging in Drebin paper
]

# handle the restricted APIs
dir_path = path.dirname(path.realpath(__file__))
dir_to_axplorer_permissions_mp = path.join(dir_path + '/res/permissions/')
txt_file_paths = list(retrive_files_set(dir_to_axplorer_permissions_mp, '', 'txt'))
sensitive_apis = []
for txt_file_path in txt_file_paths:
    file_name = path.basename(txt_file_path)
    if 'cp-map' in file_name:
        text_lines = read_txt(txt_file_path)
        for line in text_lines:
            api_name = line.split(' ')[0].strip()
            class_name, method_name = api_name.rsplit('.', 1)
            api_name_smali = java_class_name2smali_name(class_name) + '->' + method_name
            sensitive_apis.append(api_name_smali)
    else:
        text_lines = read_txt(txt_file_path)
        for line in text_lines:
            api_name = line.split('::')[0].split('(')[0].strip()
            class_name, method_name = api_name.rsplit('.', 1)
            api_name_smali = java_class_name2smali_name(class_name) + '->' + method_name
            sensitive_apis.append(api_name_smali)

path_to_lib_type_1 = path.join(dir_path + '/res/liblist_threshold_10.txt')
Third_part_libraries = ['L' + lib_cnt.split(';')[0].strip('"').lstrip('/') for lib_cnt in read_txt(
    path_to_lib_type_1, mode='r')]

paths_to_lib_type2 = retrive_files_set(dir_path + '/res/libraries', '', 'txt')
for p in paths_to_lib_type2:
    Third_part_libraries.extend([java_class_name2smali_name(lib) for lib in read_txt(p, 'r')])

TAG_SPLITTER = '#.tag#'


# 这个包装函数的作用是在并行处理过程中捕获并记录异常，
# 以便在后续步骤中处理（例如，通过在 feature_extraction 方法中检查返回值是否为异常）。
# 这些异常通常包括文件处理错误、资源不足等问题。
def apk2feat_wrapper(kwargs):
    try:
        return apk2features(*kwargs)
    except Exception as e:
        return e


def apk2features(apk_path, max_number_of_smali_files=10000, saving_path=None):
    """
    Extract the apk features, including dangerous permissions, suspicious intent actions, restricted apis, and dangerous apis
    Each permission: android permission
    Each intent action: intent action + '#.tag.#' + info of this intent
    Each api: 'invoke-type + ' ' + class_name + '->' + method_name + arguments + return_type+'#.tag.#'+ info of its
              class name and method definition path'

    :param apk_path: string, a path directs to an apk file, and otherwise an error is raised (指向一个apk文件的数量)
    :param max_number_of_smali_files: integer, the maximum number of smali files (smail文件的最大数量)
    :param timeout: integer, the elapsed time in minutes (耗时、以分钟为单位)
    :param saving_path: string, a path directs to saving path (保存路径)
    
    从APK文件中提取特征，包括危险的权限、可疑的意图操作、限制性API和危险API。
    """

    # 检查输入参数的合法性
    if not isinstance(apk_path, str):
        raise ValueError("Expected a path, but got {}".format(type(apk_path)))

    if not path.exists(apk_path):
        raise FileNotFoundError("Cannot find an apk file by following the path {}.".format(apk_path))

    if saving_path is None:
        warnings.warn("Save the features in current direction:{}".format(getcwd()))
        saving_path = path.join(getcwd(), 'api-graph')

    # 读取并分析APK文件
    try:
        apk_path = path.abspath(apk_path)
        a, d, dx = AnalyzeAPK(apk_path)  # a: app; d: dex; dx: analysis of dex
    except Exception as e:
        raise ValueError("Fail to read and analyze the apk {}:{} ".format(apk_path, str(e)))

    # 1. 获取权限
    try:
        permission_list = get_permissions(a)
    except Exception as e:
        raise ValueError("Fail to extract permissions {}:{} ".format(apk_path, str(e)))

    # 2. 获取组件（除了提供者）
    try:
        component_list = get_components(a)
    except Exception as e:
        raise ValueError("Fail to extract components {}:{} ".format(apk_path, str(e)))

    # 3. 获取提供者
    try:
        provider_list = get_providers(a)
    except Exception as e:
        raise ValueError("Fail to extract providers {}:{} ".format(apk_path, str(e)))

    # 4. 获取意图操作
    try:
        intent_actions = get_intent_actions(a)
    except Exception as e:
        raise ValueError("Fail to extract intents {}:{} ".format(apk_path, str(e)))

    # 5. 获取硬件
    try:
        hardware_list = get_hardwares(a)
    except Exception as e:
        raise ValueError("Fail to extract hardware {}:{} ".format(apk_path, str(e)))

    # 6. 获取API
    try:
        api_sequences = get_apis(d, max_number_of_smali_files)
    except Exception as e:
        raise ValueError("Fail to extract apis {}:{} ".format(apk_path, str(e)))

    # 将提取到的所有特征组合在一起
    features = []
    features.extend(permission_list + component_list + provider_list + intent_actions + hardware_list)
    features.extend(api_sequences)

    # 将特征保存到磁盘
    save_to_disk(features, saving_path)

    # 如果没有找到特征，发出警告
    if len(features) <= 0:
        warnings.warn("No features found: " + apk_path)

    return saving_path


def permission_check(permission):
    if permission in DANGEROUS_PERMISSION_TAGS:
        return True
    else:
        return False


def get_permissions(app):
    """
    Get permissions
    :param app: androidguard.core.bytecodes.apk
    """
    rtn_permssions = []
    # 获取 APK 文件中的自有权限
    permissions = app.get_permissions() 
    # 获取第三方请求的权限
    permissions += app.get_requested_third_party_permissions()
    for perm in permissions:
        rtn_permssions.append(perm + TAG_SPLITTER + PERMISSION)
    return rtn_permssions


def get_components(app):
    """
    Get activities
    :param app: androidguard.core.bytecodes.apk
    """
    component_names = []
    manifest_xml = app.get_android_manifest_xml()
    xml_dom = minidom.parseString(etree.tostring(manifest_xml, pretty_print=True))
    
    activity_elements = xml_dom.getElementsByTagName(ACTIVITY)
    for activity in activity_elements:
        if activity.hasAttribute("android:name"):
            activity_name = activity.getAttribute("android:name")
            component_names.append(activity_name + TAG_SPLITTER + ACTIVITY)
            
    service_elements = xml_dom.getElementsByTagName(SERVICE)
    for service in service_elements:
        if service.hasAttribute("android:name"):
            svc_name = service.getAttribute("android:name")
            component_names.append(svc_name + TAG_SPLITTER + SERVICE)
            
    receive_elements = xml_dom.getElementsByTagName(RECEIVER)
    for receiver in receive_elements:
        if receiver.hasAttribute("android:name"):
            receiver_name = receiver.getAttribute("android:name")
            component_names.append(receiver_name + TAG_SPLITTER + RECEIVER)

    return component_names

# 函数的目的是获取 APK 文件中的内容提供者（Content Providers）信息，并将这些内容提供者添加到一个名为 providers 的列表中。
def get_providers(app):
    providers = []
    manifest_xml = app.get_android_manifest_xml()
    xml_dom = minidom.parseString(etree.tostring(manifest_xml, pretty_print=True))
    providers_elements = xml_dom.getElementsByTagName("provider")
    
    for provider in providers_elements:
        if provider.hasAttribute("android:name"):
            prov_name = provider.getAttribute("android:name")
            writer = io.StringIO()
            provider.writexml(writer)
            prov_extra_info = writer.getvalue()
            providers.append(prov_name + TAG_SPLITTER + PROVIDER + TAG_SPLITTER + prov_extra_info)
    return providers


def intent_action_check(action_in_question):
    for pre_action_name in INTENT_TAGS:
        if pre_action_name in action_in_question:
            return True
    else:
        return False


def get_intent_actions(app):
    """
    get intent actions from manifest, and the strings of intent actions in dex code will be neglected
    intent action starts with either of
    "android.intent.action",
    "com.android.vending",
    "android.net", and
    "com.android"
    
    :param app: androidguard.core.bytecodes.apk
    """
    actions = []

    manifest_xml = app.get_android_manifest_xml()
    xml_dom = minidom.parseString(etree.tostring(manifest_xml, pretty_print=True))

    def _analyze_component(component_elements, component_name):
        for element in component_elements:
            intent_filter_elements = element.getElementsByTagName('intent-filter')
            for intent_element in intent_filter_elements:
                # Handling the intent-filters that may have multiple actions
                action_elements = intent_element.getElementsByTagName('action')
                for i, action_element in enumerate(action_elements):
                    if action_element.hasAttribute("android:name"):
                        action = action_element.getAttribute("android:name")
                        # action_parent = copy.copy(intent_element)
                        # if len(action_elements) > 1:
                        #     action_elements_copy = action_parent.getElementsByTagName('action')
                        #     for _i in range(len(action_elements_copy)):
                        #         if _i != i:
                        #             action_parent.removeChild(action_elements_copy[_i])
                        # writer = io.StringIO()
                        # action_parent.writexml(writer)
                        # action_extra_info = writer.getvalue()
                        actions.append(action + TAG_SPLITTER + INTENT + TAG_SPLITTER + component_name)

    # 1. activities
    activity_elements = xml_dom.getElementsByTagName('activity')
    _analyze_component(activity_elements, 'activity')
    # 2. services
    service_elements = xml_dom.getElementsByTagName('service')
    _analyze_component(service_elements, 'service')
    # 3. broadcast receiver
    receiver_elements = xml_dom.getElementsByTagName('receiver')
    _analyze_component(receiver_elements, 'receiver')
    return actions


def get_hardwares(app):
    hardwares_rtn = []
    for hardware in app.get_features():
        hardwares_rtn.append(hardware + TAG_SPLITTER + HARDWARE)
    return hardwares_rtn


def check_suspicious_api(api_query):
    for specific_api in DANGEROUS_API_SIMLI_TAGS:
        if specific_api in api_query:
            return True
    else:
        return False


def check_sensitive_api(api_query):
    if api_query in sensitive_apis:
        return True
    else:
        return False


def get_apis(dexes, max_number_of_smali_files):
    """
    get api sequences by class-wise
    """
    if isinstance(dexes, DalvikVMFormat):
        dexes = [dexes]

    apis_classwise = []
    for dex in dexes:
        smali_classes = dex.get_classes()
        for smali_cls in smali_classes:
            if len(apis_classwise) > max_number_of_smali_files:
                return apis_classwise # ???
            smali_methods = smali_cls.get_methods()
            apis = []
            for smali_mth in smali_methods:
                # class name + TAG_SPLITTER + .method + Modifier + method name + parameter type + return value type
                # 遍历每个方法，并构造方法的头部信息（method_header）。
                method_header = smali_mth.class_name + TAG_SPLITTER
                method_header += '.method ' + smali_mth.access_flags_string + ' ' + smali_mth.name + smali_mth.proto
                # 对于每个方法，获取其指令 smali_mth.get_instructions()
                for instruction in smali_mth.get_instructions():
                    smali_code = instruction.get_name() + ' { ' + instruction.get_output()  # on win32 platform, 'instruction.get_output()' triggers the memory exception 'exit code -1073741571 (0xC00000FD)' sometimes
                    # 如果找到调用，提取相关信息（如调用类型、类名、方法名等）
                    invoke_match = re.search(
                        r'^([ ]*?)(?P<invokeType>invoke\-([^ ]*?)) {(?P<invokeParam>([vp0-9,. ]*?)),? (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeParamType>(.*?))\)(?P<invokeReturnType>(.*?))$',
                        smali_code)
                    if invoke_match is None:
                        continue
                    # 调用类型
                    invoke_type = invoke_match.group('invokeType')
                    # 类名、方法名
                    class_name, method_name = invoke_match.group('invokeObject'), invoke_match.group('invokeMethod')
                    proto = '(' + invoke_match.group('invokeParamType') + ')' + invoke_match.group('invokeReturnType')
                    
                    # todo: justify the method is not the type of overload
                    # note: androidguard provides the EncodedMethod type, which is indeed not helpful, sometimes it is problematic
                    # e.g., from now on (version 3.3.5), the encodedmethod is actually implemented in the parent class yet neglected by androidguard
                    
                    # 检查调用的 API 是否是敏感 API（check_sensitive_api）或可疑 API（check_suspicious_api）。
                    # 如果是，则将 API 信息添加到 apis 列表中。
                    if check_sensitive_api(class_name + '->' + method_name) or \
                            check_suspicious_api(class_name + '->' + method_name):
                        api_info = invoke_type + ' ' + class_name + '->' + method_name + proto + \
                                   TAG_SPLITTER + SYS_API + \
                                   TAG_SPLITTER + method_header
                        apis.append(api_info)
            if len(apis) <= 0:
                continue
            apis_classwise.append(apis)
    return apis_classwise


def save_to_disk(data, saving_path):
    dump_pickle(data, saving_path)


def read_from_disk(loading_path):
    return read_pickle(loading_path)

'''
    这段代码定义了一个名为 get_feature_list 的函数，
    它接受一个特征列表作为输入参数，然后将其拆分为特征列表、特征信息列表和特征类型列表。
'''
def get_feature_list(feature):
    if not isinstance(feature, list):
        raise TypeError("Expect a list or nested list, but got {}.".format(type(feature)))

    feature_list = []
    feature_info_list = []
    feature_type_list = []
    
    for feat in feature:
        if isinstance(feat, str):  # manifest features
            feature_elements = feat.split(TAG_SPLITTER)
            _feature = feature_elements[0]
            _feature_type = feature_elements[1]
            if len(feature_elements) >= 3:
                _feat_info = TAG_SPLITTER.join(feature_elements[2:])
            else:
                _feat_info = ''
            feature_list.append(_feature)
            feature_type_list.append(_feature_type)
            feature_info_list.append(_feat_info)
            
        elif isinstance(feat, list):  # APIs
            for api in feat:
                feature_elements = api.split(TAG_SPLITTER)
                _api_name = get_api_name(feature_elements[0])
                feature_list.append(_api_name)
                feature_type_list.append(feature_elements[1])
                feature_info_list.append(feature_elements[0])
        else:
            raise ValueError("Expect String or List, but got {}.".format(type(feat)))
    return feature_list, feature_info_list, feature_type_list


def get_api_name(api_info):
    if not isinstance(api_info, str):
        raise TypeError
    # 使用正则表达式搜索 api_info 字符串以提取 API 调用类型、调用对象、调用方法、调用参数和调用返回值。
    invoke_match = re.search(
        r'^([ ]*?)(?P<invokeType>invoke\-([^ ]*?)) (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
        api_info)
    # 将提取到的调用对象和调用方法组合起来，以 '->' 分隔，生成 API 名称。
    _api_name = invoke_match.group('invokeObject') + '->' + invoke_match.group('invokeMethod')
    return _api_name


def get_api_info(node_tag):
    if not isinstance(node_tag, str):
        raise TypeError
    assert TAG_SPLITTER in node_tag
    api_info = node_tag.split(TAG_SPLITTER)[0]
    return api_info

# 这个 format_feature 函数的目的是对给定的特征列表进行处理
# 并将其分成两个列表：non_api_feature_list 和 api_feature_list
def format_feature(feature):
    # 检查输入的 feature 是否是列表类型，如果不是，抛出一个 TypeError 异常。
    if not isinstance(feature, list):
        raise TypeError("Expect a list or nested list, but got {}.".format(type(feature)))
    
    # 初始化两个空列表：non_api_feature_list 和 api_feature_list。
    non_api_feature_list = []
    api_feature_list = []
    for feat in feature:
        # 对于 feat 类型为字符串（即 manifest 特征）的情况：
        # a. 如果字符串中包含 TAG_SPLITTER（预定义的分隔符），则将其分割为两部分，并仅保留第一部分。
        # b. 否则，保留原始字符串。
        # c. 将处理后的字符串添加到 non_api_feature_list 中。
        if isinstance(feat, str):  # manifest features
            if TAG_SPLITTER in feat:
                _feat, _1 = feat.split(TAG_SPLITTER, 1)
            else:
                _feat = feat
            non_api_feature_list.append(_feat)
        # 对于 feat 类型为列表（即 API 列表）的情况：
        # a. 遍历包含 API 信息的 feat 列表。
        # b. 将 API 信息与其他信息分割，并调用 get_api_name 函数获取 API 名称
        # c. 将 API 名称添加到 api_feature_list 中。
        elif isinstance(feat, list):  # apis
            for api in feat:
                api_info, _1 = api.split(TAG_SPLITTER, 1)
                _api_name = get_api_name(api_info)
                api_feature_list.append(_api_name)
        else:
            raise ValueError("Expect String or List, but got {}.".format(type(feat)))
    return non_api_feature_list, api_feature_list


def get_api_class(node_tag):
    if not isinstance(node_tag, str):
        raise TypeError
    assert TAG_SPLITTER in node_tag
    api_info = node_tag.split(TAG_SPLITTER)[0]
    return api_info.split('->')[0].split(' ')[1].strip()


def get_caller_info(node_tag):
    if not isinstance(node_tag, str):
        raise TypeError
    assert TAG_SPLITTER in node_tag
    caller_info = node_tag.split(TAG_SPLITTER)[1]
    class_name, method_statement = caller_info.split(';', 1)
    # tailor tab issue that may be triggered by encodedmethod of androidguard
    method_match = re.match(
        r'^([ ]*?)\.method\s+(?P<methodPre>([^ ].*?))\((?P<methodArg>(.*?))\)(?P<methodRtn>(.*?))$', method_statement)
    method_statement = '.method ' + method_match['methodPre'].strip() + '(' + method_match['methodArg'].strip().replace(
        ' ', '') + ')' + method_match['methodRtn'].strip()
    return class_name + ';', method_statement


def get_api_tag(api_ivk_line, api_callee_class_name, api_callee_name):
    return api_ivk_line + TAG_SPLITTER + api_callee_class_name + api_callee_name


def get_same_class_prefix(entry_node_list):
    assert isinstance(entry_node_list, list)
    if len(entry_node_list) <= 0:
        return ''
    class_names = [a_node.split('.method')[0].rsplit('$')[0] for a_node in entry_node_list]
    a_class_name = class_names[0]
    n_names = a_class_name.count('/')
    for idx in range(n_names):
        pre_class_name = a_class_name.rsplit('/', idx)[0]
        if all([pre_class_name in class_name for class_name in class_names]):
            return pre_class_name
    else:
        return ''


# def _main():
#     rtn_str = apk2features(
#         '/mnt/c/Users/lideq/datasets/androzoo/178851ecaa7a21145da353489f924277e9f62ac476d3fdce786946ae8c4c9baf',
#         200000,
#         "./abc.feat")
#     print(rtn_str)
    
def _main():
    rtn_str = apk2features(
        '/home/chenzy/Android_Malware_Detection/pad4amd/APK_samples/base.apk',
        200000,
        "./feature_example.pkl")
    print(rtn_str)


if __name__ == "__main__":
    import sys

    sys.exit(_main())
