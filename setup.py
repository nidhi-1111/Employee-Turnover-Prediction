from setuptools import find_packages, setup


hyphen = '-e .'
def get_requirements(file_path):
    """
    This function will return the list of requirements.txt
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements= [req.replace('\n', ' ') for req in requirements]

    if hyphen in requirements:
        requirements.remove(hyphen)


setup (
name = 'Employee Turnover Prediction',
version = '0.0.1',
author = 'Nidhi',
author_email = 'nidhibodar17@gmail.com',
packages = find_packages(),
install_requires  = get_requirements('requirements.txt')

)