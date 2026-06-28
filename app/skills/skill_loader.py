# app/skills/skill_loader.py
"""
加载和解析 SKILL.md 文件
支持按需加载完整内容，用于注入 System Prompt 或转为工具
"""
import frontmatter
from pathlib import Path
from typing import Dict, List, Optional, Any

from utils.logger import logger


class SkillLoader:
    """
    负责扫描指定目录下的 SKILL.md 文件，建立索引，并提供内容加载
    """

    def __init__(self, skills_dir: str = "app/skills/md_skills"):
        self.skills_dir = Path(skills_dir)
        self.index: Dict[str, Dict[str, Any]] = {}  # name -> {description, path, frontmatter}

    def load_index(self) -> Dict[str, Dict[str, Any]]:
        """
        扫描 skills_dir 下所有子文件夹中的 SKILL.md，读取 frontmatter 建立索引
        返回索引字典，同时存储到 self.index
        """
        if not self.skills_dir.exists():
            logger.warning(f"⚠️  SKILLs 目录不存在: {self.skills_dir}")
            return {}

        self.index = {}
        for skill_folder in self.skills_dir.iterdir():
            if not skill_folder.is_dir():
                continue
            skill_file = skill_folder / "SKILL.md"
            if not skill_file.exists():
                continue

            try:
                with open(skill_file, 'r', encoding='utf-8') as f:
                    post = frontmatter.load(f)
                    name = post.get('name')
                    if not name:
                        logger.warning(f"⚠️  跳过 {skill_file}：缺少 name 字段")
                        continue

                    self.index[name] = {
                        "description": post.get('description', ''),
                        "frontmatter": post.metadata,
                        "path": str(skill_file),
                        "folder": str(skill_folder),
                        "body": post.content,  # 暂不保存全文，节省内存
                    }
            except Exception as e:
                logger.warning(f"⚠️  解析 {skill_file} 失败: {e}")

        logger.info(f"✅ 加载了 {len(self.index)} 个 MD 技能")
        return self.index

    def get_skill_info(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """获取技能索引信息（不含正文）"""
        return self.index.get(skill_name)

    def load_full_skill(self, skill_name: str) -> Optional[str]:
        """
        加载完整的 SKILL.md 内容（包括 frontmatter 和正文），用于注入 prompt
        """
        info = self.index.get(skill_name)
        if not info:
            return None
        skill_path = Path(info["path"])
        try:
            with open(skill_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"⚠️  读取 {skill_path} 失败: {e}")
            return None

    def get_all_skills(self) -> List[Dict[str, Any]]:
        """返回所有技能的索引信息（仅元数据）"""
        return list(self.index.values())

    def list_skill_names(self) -> List[str]:
        """返回所有技能名称"""
        return list(self.index.keys())