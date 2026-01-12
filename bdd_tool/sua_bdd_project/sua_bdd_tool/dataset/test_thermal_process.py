import pytest
import tempfile
import shutil
import os
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the functions to test
from bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process import (
    classify_and_move, get_color_ratios, DIR_NAMES, TH_YELLOW, TH_ORANGE, TH_RED
)


class TestGetColorRatios:
    """Test cases for get_color_ratios function"""
    
    def test_get_color_ratios_yellow_dominant(self):
        """Test when image is mostly yellow"""
        # Create a yellow image (BGR format)
        img = np.full((100, coordinaten0, 3), (0, 255, 255), dtype=np.uint8)  # Yellow in BGR
        
        r_yellow, r_orange, r_red = get_color_ratios(img)
        
        # Yellow should be dominant, others should be low
        assert r_yellow > 0.9
        assert r_orange < 0.1
        assert r_red < 0.1
        assert r_yellow + r_orange + r_red <= 1.0
    
    def test_get_color_ratios_orange_dominant(self):
        """Test when image is mostly orange"""
        # Create an orange image (BGR format - orange is between yellow and red)
        img = np.full((100, coordinaten0, 3), (0, 165, 255), dtype=np.uint8)  # Orange in BGR
        
        r_yellow, r_orange, r_red = get_color_ratios(img)
        
        # Orange should be detected (may overlap with yellow/red ranges)
        assert r_orange > 0.0
        assert r_yellow >= 0.0
        assert r_red >= 0.0
        assert r_yellow + r_orange + r_red <= 1.0
    
    def test_get_color_ratios_red_dominant(self):
        """Test when image is mostly red"""
        # Create a red image (BGR format)
        img = np.full((100, coordinaten0, 3), (0, 0, 255), dtype=np.uint8)  # Red in BGR
        
        r_yellow, r_orange, r_red = get_color_ratios(img)
        
        # Red should be dominant
        assert r_red > 0.9
        assert r_yellow < 0.1
        assert r_orange < 0.1
        assert r_yellow + r_orange + r_red <= 1.0
    
    def test_get_color_ratios_black_image(self):
        """Test with black image (no colors)"""
        img = np.zeros((100, coordinaten0, 3), dtype=np.uint8)  # Black image
        
        r_yellow, r_orange, r_red = get_color_ratios(img)
        
        # All color ratios should be 0
        assert r_yellow == 0.0
        assert r_orange == 0.0
        assert r_red == 0.0
    
    def test_get_color_ratios_white_image(self):
        """Test with white image (no saturated colors)"""
        img = np.full((100, coordinaten0, 3), (255, 255, 255), dtype=np.uint8)  # White image
        
        r_yellow, r_orange, r_red = get_color_ratios(img)
        
        # White has no saturation, so color ratios should be low
        assert r_yellow < 0.1
        assert r_orange < 0.1
        assert r_red < 0.1
    
    def test_get_color_ratios_mixed_colors(self):
        """Test with image containing multiple colors"""
        img = np.zeros((100, coordinaten0, 3), dtype=np.uint8)
        # Create horizontal stripes of different colors
        img[:25] = (0, 255, 255)  # Yellow
        img[25:50] = (0, 165, 255)  # Orange
        img[50:75] = (0, 0, 255)  # Red
        img[75:] = (0, 0, 0)  # Black
        
        r_yellow, r_orange, r_red = get_color_ratios(img)
        
        # Each color should have roughly 25% coverage
        assert 0.2 < r_yellow < 0.3
        assert 0.2 < r_orange < 0.3
        assert 0.2 < r_red < 0.3
        assert r_yellow + r_orange + r_red <= 1.0
    
    def test_get_color_ratios_invalid_image(self):
        """Test with invalid image data"""
        img = np.array([1, 2, 3])  # Invalid shape
        
        with patch('cv2.cvtColor', side_effect=Exception("CV2 error")):
            r_yellow, r_orange, r_red = get_color_ratios(img)
            
            # Should return 0,0,0 on error
            assert r_yellow == 0.0
            assert r_orange == 0.0
            assert r_red == 0.0


class TestClassifyAndMove:
    """Test cases for classify_and_move function"""
    
    @pytest.fixture
    def setup_temp_dirs(self):
        """Setup temporary directories for testing"""
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        src_dir = Path(temp_dir) / "source"
        dst_dir = Path(temp_dir) / "destination"
        
        # Create source directory
        src_dir.mkdir(parents=True, exist_ok=True)
        
        yield src_dir, dst_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def create_test_image(self, path, color_bgr, size=(100, coordinaten0)):
        """Create a test image with specified color"""
        img = np.full((size[0], size[1], 3), color_bgr, dtype=np.uint8)
        cv2.imwrite(str(path), img)
        return path
    
    def test_classify_and_move_yellow_image(self):
        """Test classification of yellow image"""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "source"
            src_dir.mkdir()
            
            # Create a yellow image
            yellow_img = src_dir / "yellow_test.jpg"
            self.create_test_image(yellow_img, (0, 255, 255))  # Yellow in BGR
            
            # Mock get_color_ratios to return high yellow ratio
            with patch('bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process.get_color_ratios') as mock_ratios:
                mock_ratios.return_value = (TH_YELLOW + 0.1, 0.0, 0.0)  # High yellow
                
                classify_and_move(str(src_dir), str(src_dir))
                
                # Check if file was moved to yellow folder
                yellow_dir = src_dir / DIR_NAMES["yellow"]
                assert yellow_dir.exists()
                assert (yellow_dir / "yellow_test.jpg").exists()
                assert not (src_dir / "yellow_test.jpg").exists()
    
    def test_classify_and_move_orange_image(self):
        """Test classification of orange image (no yellow)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "source"
            src_dir.mkdir()
            
            # Create an orange image
            orange_img = src_dir / "orange_test.jpg"
            self.create_test_image(orange_img, (0, 165, 255))  # Orange in BGR
            
            # Mock get_color_ratios to return moderate orange ratio, low yellow
            with patch('bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process.get_color_ratios') as mock_ratios:
                mock_ratios.return_value = (TH_YELLOW - 0.1, TH_ORANGE + 0.1, 0.0)  # Low yellow, high orange
                
                classify_and_move(str(src_dir), str(src_dir))
                
                # Check if file was moved to orange folder
                orange_dir = src_dir / DIR_NAMES["orange"]
                assert orange_dir.exists()
                assert (orange_dir / "orange_test.jpg").exists()
                assert not (src_dir / "orange_test.jpg").exists()
    
    def test_classify_and_move_red_image(self):
        """Test classification of red image (no yellow or orange)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "source"
            src_dir.mkdir()
            
            # Create a red image
            red_img = src_dir / "red_test.jpg"
            self.create_test_image(red_img, (0, 0, 255))  # Red in BGR
            
            # Mock get_color_ratios to return high red ratio, low yellow/orange
            with patch('bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process.get_color_ratios') as mock_ratios:
                mock_ratios.return_value = (TH_YELLOW - 0.1, TH_ORANGE - 0.1, TH_RED + 0.1)
                
                classify_and_move(str(src_dir), str(src_dir))
                
                # Check if file was moved to red folder
                red_dir = src_dir / DIR_NAMES["red"]
                assert red_dir.exists()
                assert (red_dir / "red_test.jpg").exists()
                assert not (src_dir / "red_test.jpg").exists()
    
    def test_classify_and_move_dark_image(self):
        """Test classification of dark image (no significant colors)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "source"
            src_dir.mkdir()
            
            # Create a dark image
            dark_img = src_dir / "dark_test.jpg"
            self.create_test_image(dark_img, (0, 0, 0))  # Black image
            
            # Mock get_color_ratios to return low ratios for all colors
            with patch('bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process.get_color_ratios') as mock_ratios:
                mock_ratios.return_value = (TH_YELLOW - 0.1, TH_ORANGE - 0.1, TH_RED - 0.1)
                
                classify_and_move(str(src_dir), str(src_dir))
                
                # Check if file was moved to dark folder
                dark_dir = src_dir / DIR_NAMES["dark"]
                assert dark_dir
                assert (dark_dir / "dark_test.jpg").exists()
                assert not (src_dir / "dark_test.jpg").exists()
    
    def test_classify_and_move_priority_yellow_over_orange(self):
        """Test that yellow has priority over orange when both are above thresholds"""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "source"
            src_dir.mkdir()
            
            test_img = src_dir / "test.jpg"
            self.create_test_image(test_img, (0, 255, 255))
            
            # Mock to return high yellow AND high orange
            with patch('bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process.get_color_ratios') as mock_ratios:
                mock_ratios.return_value = (TH_YELLOW + 0.1, TH_ORANGE + 0.1, 0.0)
                
                classify_and_move(str(src_dir), str(src_dir))
                
                # Should go to yellow folder (higher priority)
                yellow_dir = src_dir / DIR_NAMES["yellow"]
                assert (yellow_dir / "test.jpg").exists()
                orange_dir = src_dir / DIR_NAMES["orange"]
                assert not (orange_dir / "test.jpg").exists()
    
    def test_classify_and_move_multiple_files(self):
        """Test classification with multiple image files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "source"
            src_dir.mkdir()
            
            # Create multiple test images
            images = [
                ("yellow.jpg", (TH_YELLOW + 0.1, 0.0, 0.0)),
                ("orange.jpg", (TH_YELLOW - 0.1, TH_ORANGE + 0.1, 0.0)),
                ("red.jpg", (TH_YELLOW - 0.1, TH_ORANGE - 0.1, TH_RED + 0.1)),
                ("dark.jpg", (TH_YELLOW - 0.1, TH_ORANGE - 0.1, TH_RED - 0.1))
            ]
            
            for filename, _ in images:
                self.create_test_image(src_dir / filename, (100, 100, 100))
            
            # Mock get_color_ratios to return different values for different files
            def mock_ratios_side_effect(img):
                filename = Path(cv2.imread.__wrapped__(str(src_dir / "temp"))).name
                for fname, ratios in images:
                    if filename == fname:
                        return ratios
                return (0, 0, 0)
            
            with patch('bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process.get_color_ratios') as mock_ratios:
                mock_ratios.side_effect = mock_ratios_side_effect
                
                classify_and_move(str(src_dir), str(src_dir))
                
                # Check that files are in correct folders
                for filename, _ in images:
                    if "yellow" in filename:
                        assert (src_dir / DIR_NAMES["yellow"] / filename).exists()
                    elif "orange" in filename:
                        assert (src_dir / DIR_NAMES["orange"] / filename).exists()
                    elif "red" in filename:
                        assert (src_dir / DIR_NAMES["red"] / filename).exists()
                    elif "dark" in filename:
                        assert (src_dir / DIR_NAMES["dark"] / filename).exists()
    
    def test_classify_and_move_invalid_image(self):
        """Test handling of invalid/corrupted images"""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "source"
            src_dir.mkdir()
            
            # Create a non-image file
            invalid_file = src_dir / "invalid.txt"
            invalid_file.write_text("This is not an image")
            
            # Create a valid image
            valid_img = src_dir / "valid.jpg"
            self.create_test_image(valid_img, (0, 255, 255))
            
            # Mock get_color_ratios for valid image
            with patch('bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process.get_color_ratios') as mock_ratios:
                mock_ratios.return_value = (TH_YELLOW + 0.1, 0.0, 0.0)
                
                classify_and_move(str(src_dir), str(src_dir))
                
                # Valid image should be moved, invalid file should remain
                yellow_dir = src_dir / DIR_NAMES["yellow"]
                assert (yellow_dir / "valid.jpg").exists()
                assert (src_dir / "invalid.txt").exists()  # Should not be moved
    
    def test_classify_and_move_file_move_error(self):
        """Test handling of file move errors"""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "source"
            src_dir.mkdir()
            
            test_img = src_dir / "test.jpg"
            self.create_test_image(test_img, (0, 255, 255))
            
            # Mock shutil.move to raise an exception
            with patch('bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process.get_color_ratios') as mock_ratios, \
                 patch('bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process.shutil.move') as mock_move:
                
                mock_ratios.return_value = (TH_YELLOW + 0.1, 0.0, 0.0)
                mock_move.side_effect = Exception("Move failed")
                
                # Should not raise exception, should continue processing
                classify_and_move(str(src_dir), str(src_dir))
                
                # File should still be in source directory (move failed)
                assert (src_dir / "test.jpg").exists()
    
    def test_classify_and_move_empty_directory(self):
        """Test behavior with empty directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "source"
            src_dir.mkdir()
            
            # Should not raise any errors
            classify_and_move(str(src_dir), str(src_dir))
            
            # All target directories should be created
            for dir_name in DIR_NAMES.values():
                assert (src_dir / dir_name).exists()
    
    def test_classify_and_move_different_image_formats(self):
        """Test with different image file formats"""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = Path(temp_dir) / "source"
            src_dir.mkdir()
            
            # Create images in different formats
            formats = ["jpg", "jpeg", "png", "bmp", "JPG", "PNG"]
            for ext in formats:
                img_path = src_dir / f"test.{ext}"
                self.create_test_image(img_path, (0, 255, 255))
            
            # Mock get_color_ratios for all images
            with patch('bdd_tool.sua_bdd_project.sua_bdd_tool.dataset.thermal_process.get_color_ratios') as mock_ratios:
                mock_ratios.return_value = (TH_YELLOW + 0.1, 0.0, 0.0)
                
                classify_and_move(str(src_dir), str(src_dir))
                
                # All images should be moved to yellow folder
                yellow_dir = src_dir / DIR_NAMES["yellow"]
                for ext in formats:
                    assert (yellow_dir / f"test.{ext}").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])