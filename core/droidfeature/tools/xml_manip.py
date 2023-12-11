"""
feasible manipulations on androidmanifest.xml files: codes are from: https://github.com/deqangss/adv-dnn-ens-malware
"""

import os
import sys

import xml.etree.ElementTree as ET
from tools.utils import *

NAMESPACE = '{http://schemas.android.com/apk/res/android}'
ET.register_namespace('android', "http://schemas.android.com/apk/res/android")


def get_xmltree_by_ET(xml_path):
    """
    read the manifest.xml
    :param disassembly_root: the root folder
    :return: ET element
    """
    try:
        if os.path.isfile(xml_path):
            with open(xml_path, 'rb') as fr:
                parser = ET.XMLParser(encoding="utf-8")
                return ET.parse(fr, parser=parser)
        else:
            raise FileNotFoundError("Error: No such file '{}'.".format(xml_path))
    except IOError:
        raise IOError("Unable to load xml file from {}".format(xml_path))


def insert_perm_manifest(manifest_ET_tree, feature_type, spec_name, mod_count=1):
    """
    insert permission or uses-feature into AndroidManifest.xml
    """
    root = manifest_ET_tree.getroot()
    all_elems = root.findall(feature_type)

    all_spec_names = [elem.get(NAMESPACE + "name") for elem in all_elems]

    if spec_name in all_spec_names:
        MSG = 'Repetition allowed:{}/\'{}\'.'.format(feature_type, spec_name)
        for t in range(mod_count):
            ET.SubElement(root, feature_type).set(NAMESPACE + "name", spec_name)
        return MSG, False, manifest_ET_tree

    for t in range(mod_count):
        ET.SubElement(root, feature_type).set(NAMESPACE + "name", spec_name)
    MSG = "Component inserted Successfully."
    return MSG, True, manifest_ET_tree


def insert_comp_manifest(manifest_ET_tree, comp_type, comp_spec_name, mod_count=1):
    """
    insert a component into manifest.xml
    :param manifest_ET_tree: manifest ElementTree element
    :param comp_type: component types
    :param comp_spec_name: component name
    :return: info, True or False, ET tree of manifest
    """
    root = manifest_ET_tree.getroot()
    application = root.find("application")
    if application is None:
        application = ET.SubElement(root, "application")
    comp_elems = application.findall(comp_type)

    comp_names = [elem.get(NAMESPACE + "name") for elem in comp_elems]

    if comp_spec_name in comp_names:
        MSG = 'Repetition allowed:{}/\'{}\'.'.format(comp_type, comp_spec_name)
        for t in range(mod_count):
            ET.SubElement(application, comp_type).set(NAMESPACE + "name", comp_spec_name)
        return MSG, False, manifest_ET_tree
    for t in range(mod_count):
        ET.SubElement(application, comp_type).set(NAMESPACE + "name", comp_spec_name)
    MSG = "Component inserted Successfully."
    return MSG, True, manifest_ET_tree


def insert_provider_manifest(manifest_ET_tree, provider_info, mod_count=1):
    root = manifest_ET_tree.getroot()
    application = root.find("application")
    if application is None:
        application = ET.SubElement(root, "application")
        application.text = ' '

    root_str = ET.tostring(root, encoding='utf-8', method='xml').decode('utf-8')
    root_split = root_str.split('</application>')
    for _ in range(mod_count):
        root_split[0] += '\n' + provider_info + '\n'
    root_str = '</application>'.join(root_split)
    manifest_ET_back = ET.ElementTree(ET.fromstring(root_str))
    MSG = "Provider inserted Successfully."
    return MSG, True, manifest_ET_back


def insert_intent_manifest(manifest_ET_tree, comp_type, intent_spec_name, mod_count=1):
    """
    insert an intent-filter into a component of manifest.xml
    :param manifest_ET_tree: manifest ElementTree element
    :param comp_type: component types
    :param intent_spec_name: intent-filter action name
    :return: info, True or False, ET tree of manifest
    """
    root = manifest_ET_tree.getroot()
    application = root.find("application")
    if application == None:
        application = ET.SubElement(root, "application")
    comp_elems = application.findall(comp_type)

    comp_names = [elem.get(NAMESPACE + "name") for elem in comp_elems]

    comp_spec_name = 'abc'
    count = 0
    while count <= 1000:
        rdm_seed = random.randint(1, 23456)
        random.seed(rdm_seed)
        comp_spec_name = random_string(intent_spec_name) + random_name(random.randint(1, 23456)) + random_name(
            random.randint(1, 23456))
        count = count + 1
        if comp_spec_name not in comp_names:
            break

    comp_tree = ET.SubElement(application, comp_type)
    comp_tree.set(NAMESPACE + "name", comp_spec_name)
    for t in range(mod_count):
        intent_tree = ET.SubElement(comp_tree, 'intent-filter')
        ET.SubElement(intent_tree, "action").set(NAMESPACE + "name", intent_spec_name)
    MSG = "intent-filter inserted Successfully."
    return MSG, True, manifest_ET_tree


def insert_elem_manifest(manifest_ET_tree, elem_type, elem_spec_name, mod_count=1):
    root = manifest_ET_tree.getroot()
    all_elems = root.findall(elem_type)

    elem_names = [elem.get(NAMESPACE + "name") for elem in all_elems]

    if elem_spec_name in elem_names:
        for t in range(mod_count):
            ET.SubElement(root, elem_type).set(NAMESPACE + "name", elem_spec_name)

    for t in range(mod_count):
        ET.SubElement(root, elem_type).set(NAMESPACE + "name", elem_spec_name)
    return manifest_ET_tree


def get_package_name(manifest_path):
    manifest_tree = get_xmltree_by_ET(manifest_path)
    return manifest_tree.getroot().get('package')


def dump_xml(save_path, et_tree):
    try:
        if os.path.isfile(save_path):
            ET.register_namespace('android', "http://schemas.android.com/apk/res/android")
            with open(save_path, "wb") as fw:
                et_tree.write(fw, encoding="UTF-8", xml_declaration=True, method='xml')
    except Exception as ex:
        raise IOError("Unable to dump xml file {}:{}.".format(save_path, str(ex)))


def fix_invalid_id(comp_name, spec_chr='@&'):
    comp_name = comp_name.replace('$;', spec_chr+';')
    comp_name = comp_name.replace('$/', spec_chr+'/')
    if comp_name[-1] == '$':
        comp_name = comp_name[:-1] + spec_chr
    while comp_name.find('$'+spec_chr) != -1:
        comp_name = comp_name.replace('$'+spec_chr, spec_chr+spec_chr)
    return comp_name


def defix_invalid_id(comp_name,spec_chr='@&'):
    return comp_name.replace(spec_chr, '$')


def check_comp_name(manifest_ET_tree, comp_type, comp_spec_name):
    root = manifest_ET_tree.getroot()
    application = root.find("application")
    if application is None:
        return False

    comp_elems = application.findall(comp_type)  # Neglect the element under other components

    if len(comp_elems) == 0:
        return False

    comp_names = [act.get(NAMESPACE + "name") for act in comp_elems]
    if comp_spec_name not in comp_names:  # fail get component name if the component is in another components
        return False
    else:
        return True

def rename_comp_manifest(manifest_ET_tree, comp_type, comp_spec_name):
    """
    rename a component of manifest.xml
    :param manifest_ET_tree: manifest_ET_tree: manifest ElementTree element
    :param comp_type: comp_type: component types
    :param comp_spec_name: comp_spec_name: component name
    :return: info, True or False, new_name, ET tree of manifest
    """
    root = manifest_ET_tree.getroot()
    application = root.find("application")
    if application is None:
        info = "No such component '{}'\n".format(comp_type)
        return info, False, None, manifest_ET_tree

    comp_elems = application.findall(comp_type)  # Neglect the element under other components

    if len(comp_elems) == 0:
        info = "No such component name '{}'\n".format(comp_spec_name)
        return info, False, None, manifest_ET_tree

    comp_names = [act.get(NAMESPACE + "name") for act in comp_elems]
    if comp_spec_name not in comp_names:  # fail get component name if the component is in another components
        info = "No such component name '{}'\n".format(comp_spec_name)
        return info, False, None, manifest_ET_tree

    if '.' in comp_spec_name:
        comp_spec_name_split = comp_spec_name.rsplit('.', 1)
        last_name = comp_spec_name_split[-1]
        pre_name = comp_spec_name_split[0] + '.'
    else:
        last_name = comp_spec_name
        pre_name = ''

    spec_chr = '@&'
    inner_names = fix_invalid_id(last_name, spec_chr).rsplit('$', 1)
    if len(inner_names) >= 2:
        pre_name2 = inner_names[0]
        if spec_chr in inner_names[1]:
            _last_names = [crypt_identifier(n) for n in inner_names[1].split(spec_chr)]
            new_last_name = '$' + spec_chr.join(_last_names)
        else:
            new_last_name = '$' + crypt_identifier(inner_names[1])
    else:
        pre_name2 = ''
        if '@&' in inner_names[0]:
            _last_names = [crypt_identifier(n) for n in inner_names[0].split(spec_chr)]
            new_last_name = spec_chr.join(_last_names)
        else:
            new_last_name = crypt_identifier(inner_names[0])
    new_comp_name = defix_invalid_id(pre_name + pre_name2 + new_last_name)

    while comp_spec_name in comp_names:
        idx = comp_names.index(comp_spec_name)
        comp_elems[idx].set(NAMESPACE + "name", new_comp_name)
        comp_names[idx] = new_comp_name
    info = "Component renamed Successfully."
    return info, True, new_comp_name, manifest_ET_tree


def get_xml_paths(directory):
    try:
        return retrive_files_set(directory, "", ".xml")
    except IOError as ex:
        raise IOError('Failed to load xml files.')


def classname2dotstring(path_str):
    return path_str.replace('/', '.').replace(';', '')[1:]


def transform_class_name(class_names):
    for class_name in class_names:
        if class_name.startswith('L'):
            yield classname2dotstring(class_name)


def extend_name(related_class, pkg_name):
    '''extend class name based on the comp. name modes: ".name", "pkg_name.name", "name". '''
    ext = set()
    for class_name in related_class:
        if pkg_name in class_name:
            ext.add(class_name.replace(pkg_name, ""))
            ext.add(class_name.replace(pkg_name + ".", ""))
    return sorted(related_class.union(ext), key=len, reverse=True)


def change_match_xml_line(xml_line, class_strings, src_name, dst_name):
    """
    change one line in xml
    :param xml_line: a line of xml file
    :param class_strings: a set of matching string transformed from class name
    :param src_name: original comp. name
    :param dst_name: changed comp. name
    :return: changed line
    """
    for class_str in class_strings:
        # print class_str
        if class_str in xml_line:
            if '.' in class_str:
                xml_line = xml_line.replace(src_name + '"', dst_name + '"')
                xml_line = xml_line.replace(src_name + '/', dst_name + '/')
                xml_line = xml_line.replace(src_name + '>', dst_name + '>')
                xml_line = xml_line.replace(src_name + ' ', dst_name + ' ')
            else:
                xml_line = xml_line.replace('"' + src_name + '"', '"' + dst_name + '"')
                xml_line = xml_line.replace('/' + src_name + '"', '/' + dst_name + '"')
                xml_line = xml_line.replace('/' + src_name + '/', '/' + dst_name + '/')
                xml_line = xml_line.replace('/' + src_name + ' ', '/' + dst_name + ' ')
                xml_line = xml_line.replace('<' + src_name + ' ', '<' + dst_name + ' ')
                xml_line = xml_line.replace('<' + src_name + '>', '<' + dst_name + '>')
            # print src_name, dst_name

            break
    return xml_line


def change_xml(xml_paths, related_class_names, source_name, dst_name, pkg_name):
    """
    change xml files based on changed class names
    :param xml_paths: set of xml paths
    :param related_class_names: class names obtained from smali files according to the source name
    :param source_name: original class name
    :param dst_name: modified class name
    """
    related_class_transf = set(transform_class_name(related_class_names))
    related_class_ext = extend_name(related_class_transf, pkg_name)

    for xml_path in xml_paths:
        for xml_line in read_file_by_fileinput(xml_path):
            if pkg_name in xml_line or source_name in xml_line:
                xml_line = change_match_xml_line(xml_line, related_class_ext, source_name, dst_name)
            print(xml_line.strip())
