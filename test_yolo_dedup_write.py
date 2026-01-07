import os
import tempfile
import pytest
import shutil
from pathlib import Path

# 导入要测试的函数
import sys
sys.path.insert(0, 'bdd_tool/sua_data_tools')
from yolo_dedup import yolo_dedup_write

class TestYoloDedupWrite:
    """Test class for yolo_dedup_write function"""

    def setup_method(self):
        """Setup test environment before each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.dedup_label_dir = os.path.join(self.temp_dir, "dedup_labels")
        os.makedirs(self.dedup_label_dir, exist_ok=True)

    def teardown_method(self):
        """Cleanup test environment after each test"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_write_single_detection_single_image(self):
        """Test writing single detection for single image"""
        # Arrange
        yolo_group_by_id = {
            "image1": [
                {
                    "cls": 0,
                    "cxcywh": (0.5, 0.5, 0.2, 0.3),
                    "conf": 0.95,
                    "id": 1
                }
            ]
        }

        # Act
        yolo_dedup_write(yolo_group_by_id, self.dedup_label_dir)

        # Assert
        output_file = os.path.join(self.dedup_label_dir, "image1.txt")
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            content = f.read().strip()
        
        expected = "0 0.500000 0.500000 0.200000 0.300000 0.9500 1"
        assert content == expected

    def test_write_multiple_detections_single_image(self):
        """Test writing multiple detections for single image"""
        # Arrange
        yolo_group_by_id = {
            "image1": [
                {
                    "cls": 0,
                    "cxcywh": (0.1, 0.2, 0.3, 0.4),
                    "conf": 0.85,
                    "id": 1
                },
                {
                    "cls": 1,
                    "cxcywh": (0.6, 0.7, 0.1, 0.2),
                    "conf": 0.92,
                    "id": 2
                }
            ]
        }

        # Act
        yolo_dedup_write(yolo_group_by_id, self.dedup_label_dir)

        # Assert
        output_file = os.path.join(self.dedup_label_dir, "image1.txt")
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            lines = f.read().strip().split('\n')
        
        assert len(lines) == 2
        assert "0 0.100000 0.200000 0.300000 0.400000 0.8500 1" in lines
        assert "1 0.600000 0.700000 0.100000 0.200000 0.9200 2" in lines

    def test_write_multiple_images(self):
        """Test writing detections for multiple images"""
        # Arrange
        yolo_group_by_id = {
            "image1": [
                {
                    "cls": 0,
                    "cxcywh": (0.1, 0.1, 0.1, 0.1),
                    "conf": 0.99,
                    "id": 1
                }
            ],
            "image2": [
                {
                    "cls": 2,
                    "cxcywh": (0.9, 0.9, 0.05, 0.05),
                    "conf": 0.78,
                    "id": 3
                }
            ]
        }

        # Act
        yolo_dedup_write(yolo_group_by_id, self.dedup_label_dir)

        # Assert - Check both files exist
        file1 = os.path.join(self.dedup_label_dir, "image1.txt")
        file2 = os.path.join(self.dedup_label_dir, "image2.txt")
        
        assert os.path.exists(file1)
        assert os.path.exists(file2)
        
        # Check content of first file
        with open(file1, 'r') as f:
            content1 = f.read().strip()
        assert content1 == "0 0.100000 0.100000 0.100000 0.100000 0.9900 1"
        
        # Check content of second file
        with open(file2, 'r') as f:
            content2 = f.read().strip()
        assert content2 == "2 0.900000 0.900000 0.050000 0.050000 0.7800 3"

    def test_write_edge_case_coordinates(self):
        """Test writing boundary coordinate values"""
        # Arrange
        yolo_group_by_id = {
            "edge_image": [
                {
                    "cls": 3,
                    "cxcywh": (0.0, 0.0, 1.0, 1.0),  # Min values
                    "conf": 0.01,
                    "id": 0
                },
                {
                    "cls": 4,
                    "cxcywh": (0.999999, 0.999999, 0.000001, 0.000001),  # Max values
                    "conf": 0.9999,
                    "id": 255
                }
            ]
        }

        # Act
        yolo_dedup_write(yolo_group_by_id, self.dedup_label_dir)

        # Assert
        output_file = os.path.join(self.dedup_label_dir, "edge_image.txt")
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            lines = f.read().strip().split('\n')
        
        assert len(lines) == 2
        assert "3 0.000000 0.000000 1.000000 1.000000 0.0100 0" in lines
        assert "4 0.999999 0.999999 0.000001 0.000001 0.9999 255" in lines

    def test_write_different_class_ids(self):
        """Test writing detections with various class IDs"""
        # Arrange
        yolo_group_by_id = {
            "class_test": [
                {
                    "cls": -1,  # Negative class (edge case)
                    "cxcywh": (0.5, 0.5, 0.1, 0.1),
                    "conf": 0.5,
                    "id": 10
                },
                {
                    "cls": 0,   # Zero class
                    "cxcywh": (0.2, 0.2, 0.2, 0.2),
                    "conf": 0.6,
                    "id": 11
                },
                {
                    "cls": 999, # Large class ID
                    "cxcywh": (0.8, 0.8, 0.3, 0.3),
                    "conf": 0.7,
                    "id": 12
                }
            ]
        }

        # Act
        yolo_dedup_write(yolo_group_by_id, self.dedup_label_dir)

        # Assert
        output_file = os.path.join(self.dedup_label_dir, "class_test.txt")
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            lines = f.read().strip().split('\n')
        
        assert len(lines) == 3
        assert "-1 0.500000 0.500000 0.100000 0.100000 0.5000 10" in lines
        assert "0 0.200000 0.200000 0.200000 0.200000 0.6000 11" in lines
        assert "999 0.800000 0.800000 0.300000 0.300000 0.7000 12" in lines

    def test_write_precision_formatting(self):
        """Test that coordinates and confidence are formatted with correct precision"""
        # Arrange
        yolo_group_by_id = {
            "precision_test": [
                {
                    "cls": 1,
                    "cxcywh": (0.123456789, 0.987654321, 0.111111, 0.222222),
                    "conf": 0.123456789,
                    "id": 42
                }
            ]
        }

        # Act
        yolo_dedup_write(yolo_group_by_id, self.dedup_label_dir)

        # Assert
        output_file = os.path.join(self.dedup_label_dir, "precision_test.txt")
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            content = f.read().strip()
        
        # Should format to 6 decimals for coordinates, 4 for confidence
        expected = "1 0.123457 0.987654 0.111111 0.222222 0.1235 42"
        assert content == expected

    def test_write_empty_detections(self):
        """Test writing empty detection list (should create empty file)"""
        # Arrange
        yolo_group_by_id = {
            "empty_image": []
        }

        # Act
        yolo_dedup_write(yolo_group_by_id, self.dedup_label_dir)

        # Assert
        output_file = os.path.join(self.dedup_label_dir, "empty_image.txt")
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            content = f.read().strip()
        
        assert content == ""

    def test_write_to_nonexistent_directory(self):
        """Test writing to directory that doesn't exist (should create it)"""
        # Arrange
        new_dir = os.path.join(self.temp_dir, "new_subdir", "labels")
        yolo_group_by_id = {
            "test_image": [
                {
                    "cls": 5,
                    "cxcywh": (0.5, 0.5, 0.1, 0.1),
                    "conf": 0.88,
                    "id": 7
                }
            ]
        }

        # Act
        yolo_dedup_write(yolo_group_by_id, new_dir)

        # Assert
        output_file = os.path.join(new_dir, "test_image.txt")
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            content = f.read().strip()
        
        assert content == "5 0.500000 0.500000 0.100000 0.100000 0.8800 7"

    def test_write_large_number_of_detections(self):
        """Test writing large number of detections to ensure performance"""
        # Arrange
        detections = []
        for i in range(100):  # 100 detections
            detections.append({
                "cls": i % 10,
                "cxcywh": (i/100.0, i/100.0, 0.1, 0.1),
                "conf": i/100.0,
                "id": i
            })
        
        yolo_group_by_id = {"large_image": detections}

        # Act
        yolo_dedup_write(yolo_group_by_id, self.dedup_label_dir)

        # Assert
        output_file = os.path.join(self.dedup_label_dir, "large_image.txt")
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            lines = f.read().strip().split('\n')
        
        assert len(lines) == 100
        
        # Check first and last detections
        first_line = lines[0]
        last_line = lines[-1]
        
        assert "0 0.000000 0.000000 0.100000 0.100000 0.0000 0" in first_line
        assert "9 0.990000 0.990000 0.100000 0.100000 0.9900 99" in last_line

    def test_write_special_characters_in_image_name(self):
        """Test writing with special characters in image names"""
        # Arrange
        yolo_group_by_id = {
            "image_with_underscores": [
                {
                    "cls": 1,
                    "cxcywh": (0.5, 0.5, 0.1, 0.1),
                    "conf": 0.9,
                    "id": 1
                }
            ],
            "image-with-dashes": [
                {
                    "cls": 2,
                    "cxcywh": (0.6, 0.6, 0.2, 0.2),
                    "conf": 0.8,
                    "id": 2
                }
            ],
            "image with spaces": [  # This should still work
                {
                    "cls": 3,
                    "cxcywh": (0.7, 0.7, 0.3, 0.3),
                    "conf": 0.7,
                    "id": 3
                }
            ]
        }

        # Act
        yolo_dedup_write(yolo_group_by_id, self.dedup_label_dir)

        # Assert - Check all files were created
        files_created = os.listdir(self.dedup_label_dir)
        assert "image_with_underscores.txt" in files_created
        assert "image-with-dashes.txt" in files_created
        assert "image with spaces.txt" in files_created

    def test_write_overwrite_existing_file(self):
        """Test that function overwrites existing files correctly"""
        # Arrange - Create a file first
        existing_file = os.path.join(self.dedup_label_dir, "test_image.txt")
        with open(existing_file, 'w') as f:
            f.write("old content")
        
        yolo_group_by_id = {
            "test_image": [
                {
                    "cls": 1,
                    "cxcywh": (0.5, 0.5, 0.1, 0.1),
                    "conf": 0.9,
                    "id": 1
                }
            ]
        }

        # Act
        yolo_dedup_write(yolo_group_by_id, self.dedup_label_dir)

        # Assert - File should be overwritten with new content
        with open(existing_file, 'r') as f:
            content = f.read().strip()
        
        assert content == "1 0.500000 0.500000 0.100000 0.100000 0.9000 1"
        assert content != "old content"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])