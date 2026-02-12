class FileUtils:
    """
    文件工具类
    """
    @staticmethod
    def save_ply(ply_path: str, output: dict):
        """
        保存 ply 文件
        Args:
            ply_path: 保存路径
            output: 输出字典
        """
        # GaussianSplat
        output["gs"].save_ply(ply_path)
        
    @staticmethod
    def save_glb(glb_path: str, output: dict):
        """
        保存 glb 文件
        Args:
            glb_path: 保存路径
            output: 输出字典
        """
        # trimesh.base.Trimesh
        output["glb"].export(glb_path)
        
    @staticmethod
    def save_gif(gif_path: str, output: dict):
        """
        保存 gif 文件
        Args:
            gif_path: 保存路径
            output: 输出字典
        """
        pass

