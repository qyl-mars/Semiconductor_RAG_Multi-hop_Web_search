from config.configs import Config
from kb.kb_manager import get_knowledge_bases

if __name__ == "__main__":
    print("KB base dir:", Config.kb_base_dir)
    print("KB list:", get_knowledge_bases())