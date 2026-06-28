"""
技能注册中心：管理所有可用的复合技能
"""
from typing import Dict, List, Type, Optional
from app.skills.base import BaseSkill
from app.skills.builtin.vision_composer import ComprehensiveAnalysisSkill
from utils.logger import logger


class SkillRegistry:
    """单例模式，管理技能"""
    _instance = None
    _skills: Dict[str, BaseSkill] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """注册所有内置技能（可扩展为动态发现）"""
        self._skills = {}
        
        # 注册内置技能
        self.register(ComprehensiveAnalysisSkill())
        # 未来可以继续注册: self.register(DocumentParsingSkill())
        # self.register(ImageCaptioningSkill())

        logger.info(f"✅ 技能注册完成: 共 {len(self._skills)} 个技能")
        for name in self._skills:
            logger.info(f"   - {name}")

    def register(self, skill: BaseSkill):
        """注册一个技能"""
        if skill.name in self._skills:
            logger.warning(f"⚠️  技能 {skill.name} 已存在，将被覆盖")
        self._skills[skill.name] = skill

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        """根据名称获取技能"""
        return self._skills.get(name)

    def list_skills(self) -> List[str]:
        """列出所有技能名称"""
        return list(self._skills.keys())

    def get_all_skills(self) -> List[BaseSkill]:
        """获取所有技能实例"""
        return list(self._skills.values())

    def get_tools(self):
        """获取所有技能的 LangChain Tool 表示（供 Agent 使用）"""
        return [skill.to_langchain_tool() for skill in self._skills.values()]
