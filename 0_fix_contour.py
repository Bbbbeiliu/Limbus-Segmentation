import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
import argparse
from typing import List, Tuple, Dict, Any
import time


class OptimizedScleraDetector:
    """
    Optimized sclera contour concavity detector and repair tool
    This class detects and repairs concavities in sclera contours using
    curvature analysis and circular fitting techniques.
    """

    def __init__(self, min_concavity_depth: float = 3,
                 curvature_threshold: float = -0.08,
                 circularity_threshold: float = 0.85,
                 window_size: int = 7):
        """
        Initialize the detector with optimized parameters

        Args:
            min_concavity_depth: Minimum depth to consider as concavity
            curvature_threshold: Curvature threshold for concavity detection
            circularity_threshold: Threshold for circular shape quality
            window_size: Size of local analysis window
        """
        self.min_concavity_depth = min_concavity_depth
        self.curvature_threshold = curvature_threshold
        self.circularity_threshold = circularity_threshold
        self.window_size = window_size
        self._precomputed_cos = None
        self._precomputed_sin = None

    def _precompute_trigonometric_tables(self, size: int):
        """Precompute trigonometric tables for faster circle generation"""
        angles = np.linspace(0, 2 * np.pi, size)
        self._precomputed_cos = np.cos(angles)
        self._precomputed_sin = np.sin(angles)

    def load_labelme_json(self, json_path: str) -> Tuple[List[np.ndarray], Tuple[int, int]]:
        """
        Load LabelMe JSON file and extract polygon points

        Args:
            json_path: Path to JSON file

        Returns:
            Tuple of (list of point arrays, image dimensions)
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_height = data['imageHeight']
        image_width = data['imageWidth']

        all_points = []
        for shape in data['shapes']:
            if shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.float32)
                all_points.append(points)

        return all_points, (image_width, image_height)

    def calculate_circularity(self, points: np.ndarray) -> float:
        """
        Calculate circularity metric for contour

        Args:
            points: Contour points

        Returns:
            Circularity value (1.0 = perfect circle)
        """
        contour_area = cv2.contourArea(points)
        perimeter = cv2.arcLength(points, True)

        if perimeter > 0:
            return 4 * np.pi * contour_area / (perimeter * perimeter)
        return 0.0

    def fast_fit_circle(self, points: np.ndarray) -> Tuple[Tuple[float, float], float, bool]:
        """
        Fast circle fitting using algebraic method instead of iterative optimization

        Args:
            points: Points to fit circle to

        Returns:
            Tuple of (center, radius, success_flag)
        """
        x = points[:, 0]
        y = points[:, 1]

        # Algebraic circle fitting (faster than least squares)
        n = len(points)
        if n < 3:
            return (np.mean(x), np.mean(y)), np.mean(np.sqrt((x - np.mean(x)) ** 2 + (y - np.mean(y)) ** 2)), False

        try:
            # Center calculation using linear algebra
            A = np.vstack([2 * x, 2 * y, np.ones(n)]).T
            b = x ** 2 + y ** 2
            c = np.linalg.lstsq(A, b, rcond=None)[0]

            xc, yc = c[0], c[1]
            r = np.sqrt(c[2] + xc ** 2 + yc ** 2)

            return (xc, yc), r, True
        except:
            # Fallback to simple method
            xc, yc = np.mean(x), np.mean(y)
            r = np.mean(np.sqrt((x - xc) ** 2 + (y - yc) ** 2))
            return (xc, yc), r, False

    def vectorized_curvature_analysis(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized curvature analysis for faster computation

        Args:
            points: Contour points

        Returns:
            Tuple of (curvatures, concavity_scores)
        """
        n = len(points)

        # Create rolled arrays for vectorized computation
        points_rolled_neg2 = np.roll(points, -2, axis=0)
        points_rolled_pos2 = np.roll(points, 2, axis=0)

        # Vectorized curvature calculation
        vec1 = points - points_rolled_neg2
        vec2 = points_rolled_pos2 - points

        # Vectorized angle calculation
        angles1 = np.arctan2(vec1[:, 1], vec1[:, 0])
        angles2 = np.arctan2(vec2[:, 1], vec2[:, 0])

        # Vectorized angle difference with period correction
        angle_diff = angles2 - angles1
        angle_diff = np.where(angle_diff > np.pi, angle_diff - 2 * np.pi, angle_diff)
        angle_diff = np.where(angle_diff < -np.pi, angle_diff + 2 * np.pi, angle_diff)

        # Vectorized arc length calculation
        arc_lengths = (np.linalg.norm(vec1, axis=1) + np.linalg.norm(vec2, axis=1)) / 2

        # Avoid division by zero
        arc_lengths = np.where(arc_lengths == 0, 1e-10, arc_lengths)
        curvatures = angle_diff / arc_lengths

        # Vectorized concavity scoring
        concavity_scores = np.zeros(n)
        mask = curvatures < self.curvature_threshold
        concavity_scores[mask] = np.abs(curvatures[mask] - self.curvature_threshold)

        return curvatures, concavity_scores

    def fast_concavity_detection(self, points: np.ndarray, curvatures: np.ndarray,
                                 concavity_scores: np.ndarray) -> List[List[int]]:
        """
        Fast detection of continuous concavity regions

        Args:
            points: Contour points
            curvatures: Computed curvature values
            concavity_scores: Concavity scores

        Returns:
            List of concavity regions (each region is list of indices)
        """
        n = len(points)
        concavity_mask = concavity_scores > 0

        # Find regions using efficient numpy operations
        diff = np.diff(concavity_mask.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1

        regions = []
        for start, end in zip(starts, ends):
            if end - start + 1 >= 2:  # Minimum region size
                region = list(range(start, end + 1))
                regions.append(region)

        # Handle circular boundary case
        if regions and regions[0][0] == 0 and regions[-1][-1] == n - 1:
            merged = regions[-1] + regions[0]
            regions[0] = merged
            regions.pop()

        return regions

    def optimized_circle_repair(self, points: np.ndarray,
                                concavity_regions: List[List[int]]) -> np.ndarray:
        """
        Optimized circle-based repair of concavity regions

        Args:
            points: Original contour points
            concavity_regions: Detected concavity regions

        Returns:
            Repaired contour points
        """
        if not concavity_regions:
            return points.copy()

        repaired_points = points.copy()
        n = len(points)

        for region in concavity_regions:
            if len(region) < 2:
                continue

            # Get context points for circle fitting
            start_idx = (region[0] - 1) % n
            end_idx = (region[-1] + 1) % n

            # Efficient context point collection
            context_size = min(8, n // 6)  # Reduced for speed
            context_indices = set()

            # Add context points before region
            for i in range(context_size):
                idx = (start_idx - i) % n
                context_indices.add(idx)

            # Add context points after region
            for i in range(context_size):
                idx = (end_idx + i) % n
                context_indices.add(idx)

            context_points = points[list(context_indices)]

            # Fast circle fitting
            center, radius, success = self.fast_fit_circle(context_points)

            if success:
                # Precompute trigonometric tables if needed
                num_points = len(region)
                if self._precomputed_cos is None or len(self._precomputed_cos) < num_points + 2:
                    self._precompute_trigonometric_tables(num_points + 10)

                # Calculate angles for arc generation
                start_angle = np.arctan2(points[start_idx][1] - center[1],
                                         points[start_idx][0] - center[0])
                end_angle = np.arctan2(points[end_idx][1] - center[1],
                                       points[end_idx][0] - center[0])

                if end_angle < start_angle:
                    end_angle += 2 * np.pi

                # Generate arc points using precomputed tables
                angles = np.linspace(start_angle, end_angle, num_points + 2)[1:-1]

                # Vectorized point generation
                cos_vals = np.cos(angles)
                sin_vals = np.sin(angles)
                x_points = center[0] + radius * cos_vals
                y_points = center[1] + radius * sin_vals

                # Assign repaired points
                repaired_points[region] = np.column_stack([x_points, y_points])

        return repaired_points

    def process_contour_optimized(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Optimized single contour processing pipeline

        Args:
            points: Contour points to process

        Returns:
            Dictionary containing processing results
        """
        start_time = time.time()

        # Calculate circularity
        circularity = self.calculate_circularity(points)

        # Fast circle fitting
        center, radius, fit_success = self.fast_fit_circle(points)

        # Vectorized curvature analysis
        curvatures, concavity_scores = self.vectorized_curvature_analysis(points)

        # Fast concavity detection
        concavity_regions = self.fast_concavity_detection(points, curvatures, concavity_scores)

        # Optimized repair
        repaired_points = self.optimized_circle_repair(points, concavity_regions)

        # Calculate repaired circularity
        repaired_circularity = self.calculate_circularity(repaired_points)

        processing_time = time.time() - start_time

        return {
            'original_points': points,
            'repaired_points': repaired_points,
            'circularity': circularity,
            'repaired_circularity': repaired_circularity,
            'center': center,
            'radius': radius,
            'fit_success': fit_success,
            'curvature_analysis_time': processing_time,
            'concavity_regions': concavity_regions,
            'num_concavities': len(concavity_regions)
        }

    def process_single_json_optimized(self, json_path: str, output_dir: str) -> Tuple[List[Dict], Tuple[int, int]]:
        """
        Optimized processing of single JSON file

        Args:
            json_path: Path to JSON file
            output_dir: Output directory for results

        Returns:
            Tuple of (results list, image dimensions)
        """
        # Load data
        all_points, image_size = self.load_labelme_json(json_path)

        if not all_points:
            print(f"No polygon data found in {json_path}")
            return [], image_size

        # Process all contours
        results = []
        total_time = 0

        for i, points in enumerate(all_points):
            result = self.process_contour_optimized(points)
            results.append(result)
            total_time += result['curvature_analysis_time']

            print(f"Contour {i + 1}: Circularity {result['circularity']:.3f} -> {result['repaired_circularity']:.3f}")
            print(f"          Detected {result['num_concavities']} concavity regions")

        print(f"Total processing time: {total_time:.3f}s")
        return results, image_size

    def generate_efficient_output(self, json_path: str, output_dir: str,
                                  results: List[Dict], image_size: Tuple[int, int]) -> str:
        """
        Generate efficient visualization output

        Args:
            json_path: Source JSON file path
            output_dir: Output directory
            results: Processing results
            image_size: Image dimensions

        Returns:
            Path to generated output file
        """
        # Create simplified visualization (faster than detailed one)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        for i, result in enumerate(results):
            orig_points = result['original_points']
            repair_points = result['repaired_points']

            # Left: Original with concavity markers
            axes[0].plot(orig_points[:, 0], orig_points[:, 1], 'b-', linewidth=2,
                         label=f'Original (C: {result["circularity"]:.3f})')

            # Mark concavity regions
            for region in result['concavity_regions']:
                region_points = orig_points[region]
                axes[0].scatter(region_points[:, 0], region_points[:, 1],
                                c='red', s=20, marker='x', alpha=0.7)

            # Right: Repaired contour
            axes[1].plot(orig_points[:, 0], orig_points[:, 1], 'b-', linewidth=1, alpha=0.3)
            axes[1].plot(repair_points[:, 0], repair_points[:, 1], 'r-', linewidth=2,
                         label=f'Repaired (C: {result["repaired_circularity"]:.3f})')

        # Configure plots
        axes[0].set_title('Original Contour with Concavity Detection')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].set_title('Repaired Contour')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()

        # Save output
        json_name = Path(json_path).stem
        output_path = Path(output_dir) / f"{json_name}_optimized_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')  # Reduced DPI for speed
        plt.close()

        print(f"Optimized analysis saved: {output_path}")
        return str(output_path)

    def generate_binary_mask(self, json_path: str, output_dir: str,
                             results: List[Dict], image_size: Tuple[int, int]) -> str:
        """
        Generate binary mask from repaired contours

        Args:
            json_path: Source JSON file path
            output_dir: Output directory
            results: Processing results
            image_size: Image dimensions

        Returns:
            Path to generated mask file
        """
        # Create blank image
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        # Draw repaired contours
        for result in results:
            repair_points = result['repaired_points'].astype(np.int32)
            cv2.fillPoly(mask, [repair_points], 255)

        # Save mask
        json_name = Path(json_path).stem
        output_path = Path(output_dir) / f"{json_name}_repaired_mask.png"
        cv2.imwrite(str(output_path), mask)

        print(f"Binary mask saved: {output_path}")
        return str(output_path)

    def process_folder_optimized(self, input_folder: str, output_folder: str) -> List[str]:
        """
        Optimized batch processing of JSON folder

        Args:
            input_folder: Input folder containing JSON files
            output_folder: Output folder for results

        Returns:
            List of processed file paths
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        json_files = list(input_path.glob("*.json"))

        if not json_files:
            print(f"No JSON files found in {input_folder}")
            return []

        print(f"Found {len(json_files)} JSON files")

        processed_files = []
        total_start_time = time.time()

        for json_file in json_files:
            try:
                print(f"\nProcessing: {json_file.name}")
                file_start_time = time.time()

                # Process single file
                results, image_size = self.process_single_json_optimized(json_file, output_path)

                if results:
                    # Generate outputs
                    analysis_path = self.generate_efficient_output(json_file, output_path, results, image_size)
                    mask_path = self.generate_binary_mask(json_file, output_path, results, image_size)

                    processed_files.extend([analysis_path, mask_path])

                file_time = time.time() - file_start_time
                print(f"File processing time: {file_time:.3f}s")

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

        total_time = time.time() - total_start_time
        print(f"\nBatch processing completed!")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average per file: {total_time / len(json_files):.3f}s")

        return processed_files


def main():
    """Main function for command line execution"""
    parser = argparse.ArgumentParser(description='Optimized Sclera Contour Concavity Detection and Repair')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing JSON files')
    parser.add_argument('--output', '-o', required=True, help='Output folder for PNG results')

    args = parser.parse_args()

    # Create optimized detector
    detector = OptimizedScleraDetector(
        min_concavity_depth=3,
        curvature_threshold=-0.08,
        circularity_threshold=0.85,
        window_size=5  # Reduced for speed
    )

    # Process folder
    detector.process_folder_optimized(args.input, args.output)


if __name__ == "__main__":
    # Example usage
    input_folder = 'data/007'
    output_folder = 'mask/007_fixed_2'

    detector = OptimizedScleraDetector()
    detector.process_folder_optimized(input_folder, output_folder)