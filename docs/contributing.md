# Guideline for contributing

## Add dependencies

If your contributing relies on dependencies from outside (such as `matplotlib` module of python), you need to assure the dependencies are properly installed.

- For python modules that `cat` relying on, you could add it in [requirements.txt](../requirements.txt)
- Some modules have their special installation processes (like `kenlm`), then you should add the installation in [install.sh](../install.sh), where you'll modify:
   1. add the new module in `choices` list of the parser
   2. add installation process in `exc_install()`
   3. add uninstallation process in `exc_rm()`