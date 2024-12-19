from setuptools import setup, find_packages

setup(
    name='mcqgenerator',
    version='0.1',
    author='santosh kumar',
    author_email='skongonda@gmail.com',
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
)