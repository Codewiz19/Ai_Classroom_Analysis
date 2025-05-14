import os
import cv2
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import shutil

class ClassroomTimelineAnalyzer:
    """
    Class to handle classroom timeline analysis by processing multiple images
    and tracking student engagement over time.
    """
    
    def __init__(self, timeline_folder, recognition_function, process_fn):
        """
        Initialize the timeline analyzer
        
        Args:
            timeline_folder (str): Path to folder containing classroom timeline images
            recognition_function (callable): Function to perform face recognition and analysis
        """
        self.timeline_folder = timeline_folder
        self.recognition_function = recognition_function
        self.process_results_fn   = process_fn
        self.results_folder = os.path.join(timeline_folder, "analysis_results")
        self.timeline_data = []
        
        # Create results folder if it doesn't exist
        os.makedirs(self.results_folder, exist_ok=True)
    
    def _is_valid_image(self, filename):
        """Check if file is a valid image based on extension"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        ext = os.path.splitext(filename)[1].lower()
        return ext in valid_extensions
    
    def _extract_timestamp_from_filename(self, filename):
        """Extract timestamp from filename (format: YYYY-MM-DD_HH-MM-SS.jpg)"""
        try:
            # Try to extract timestamp from filename pattern
            base_name = os.path.splitext(filename)[0]
            timestamp = datetime.strptime(base_name, "%Y-%m-%d_%H-%M-%S")
            return timestamp
        except ValueError:
            # If not in expected format, use file creation time
            filepath = os.path.join(self.timeline_folder, filename)
            creation_time = os.path.getctime(filepath)
            return datetime.fromtimestamp(creation_time)
    
    def get_image_files(self):
        """Get sorted list of image files in the timeline folder"""
        if not os.path.exists(self.timeline_folder):
            return []
            
        files = [f for f in os.listdir(self.timeline_folder) 
                if os.path.isfile(os.path.join(self.timeline_folder, f)) and self._is_valid_image(f)]
        
        # Sort files by timestamp
        files.sort(key=lambda f: self._extract_timestamp_from_filename(f))
        
        return files
    
    def process_timeline(self, db_path, hopenet_model=None, transform=None, class_roster=None):
        """
        Process all images in the timeline folder and analyze student engagement
        
        Args:
            db_path (str): Path to student database for face recognition
            hopenet_model: Model for head pose estimation
            transform: Image transformation for the hopenet model
            class_roster (list): List of student names in the class
            
        Returns:
            dict: Timeline analysis results
        """
        image_files = self.get_image_files()
        
        if not image_files:
            return {"error": "No images found in timeline folder"}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        self.timeline_data = []
        
        for i, image_file in enumerate(image_files):
            progress = (i+1) / len(image_files)
            progress_bar.progress(progress)
            
            status_text.text(f"Processing image {i+1}/{len(image_files)}: {image_file}")
            
            # Open and process image
            image_path = os.path.join(self.timeline_folder, image_file)
            image = Image.open(image_path)
            
            # Get timestamp from filename
            timestamp = self._extract_timestamp_from_filename(image_file)
            
            # Process image
            result = self.recognition_function(
                image=image,
                db_path=db_path,
                hopenet_model=hopenet_model,
                transform=transform
            )
            
            # Save result to results folder
            result_filename = f"result_{os.path.splitext(image_file)[0]}.json"
            result_path = os.path.join(self.results_folder, result_filename)
            
            # Create clean version of result for saving
            save_result = result.copy()
            if 'annotated_img_rgb' in save_result:
                del save_result['annotated_img_rgb']
                
            with open(result_path, 'w') as f:
                json.dump(save_result, f, indent=2)
            
            # Save annotated image
            if 'annotated_img_rgb' in result:
                annotated_img = cv2.cvtColor(result['annotated_img_rgb'], cv2.COLOR_RGB2BGR)
                annotated_path = os.path.join(self.results_folder, f"annotated_{image_file}")
                cv2.imwrite(annotated_path, annotated_img)
            
            # Process for timeline data
            insights = self.process_results_fn(result, class_roster)
            
            # Add timestamp
            timeline_entry = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "filename": image_file,
                "insights": insights
            }
            
            self.timeline_data.append(timeline_entry)
        
        status_text.text("Timeline processing complete!")
        
        # Generate timeline analysis
        timeline_analysis = self.analyze_timeline()
        
        return timeline_analysis
    
    def analyze_timeline(self):
        """Analyze the timeline data to extract trends and patterns"""
        if not self.timeline_data:
            return {"error": "No timeline data available"}
        
        # Extract temporal data
        timestamps = []
        avg_engagement_scores = []
        attendance_counts = []
        emotion_counts = {
            "happy": [], "neutral": [], "sad": [], "angry": [], 
            "fear": [], "surprise": [], "disgust": []
        }
        
        # Student tracking
        student_engagement_history = {}
        
        for entry in self.timeline_data:
            timestamp = entry["timestamp"]
            insights = entry["insights"]
            
            # Basic metrics
            timestamps.append(timestamp)
            avg_engagement_scores.append(insights["engagement"]["average_engagement_score"])
            attendance_counts.append(len(insights["attendance"]["present_students"]))
            
            # Most common emotion
            most_common = insights["engagement"]["most_common_emotion"]
            
            # Count all emotions
            emotion_count = {emotion: 0 for emotion in emotion_counts.keys()}
            
            # Count emotions from student details
            for student in insights["student_details"]:
                # Track individual student engagement
                student_name = student["name"]
                if student_name not in student_engagement_history:
                    student_engagement_history[student_name] = []
                    
                student_engagement_history[student_name].append({
                    "timestamp": timestamp,
                    "engagement_score": student["engagement_score"],
                    "primary_emotion": student["primary_emotion"]["emotion"],
                    "attention_direction": student.get("attention_direction", "forward")
                })
                
                # Count primary emotion
                primary_emotion = student["primary_emotion"]["emotion"]
                if primary_emotion in emotion_count:
                    emotion_count[primary_emotion] += 1
            
            # Add counts to history
            for emotion, count in emotion_count.items():
                emotion_counts[emotion].append(count)
        
        # Calculate trends
        engagement_trend = self._calculate_trend(avg_engagement_scores)
        attendance_trend = self._calculate_trend(attendance_counts)
        
        # Find students with most improvement and decline
        student_trends = {}
        for student, history in student_engagement_history.items():
            if len(history) >= 2:  # Need at least 2 points for trend
                scores = [entry["engagement_score"] for entry in history]
                student_trends[student] = self._calculate_trend(scores)
        
        # Identify most improved and most declined
        if student_trends:
            most_improved = max(student_trends.items(), key=lambda x: x[1])[0]
            most_declined = min(student_trends.items(), key=lambda x: x[1])[0]
        else:
            most_improved = "N/A"
            most_declined = "N/A"
        
        # Create analysis result
        analysis = {
            "timeline_summary": {
                "num_sessions": len(self.timeline_data),
                "time_period": {
                    "start": timestamps[0] if timestamps else "N/A",
                    "end": timestamps[-1] if timestamps else "N/A"
                },
                "trends": {
                    "engagement": {
                        "direction": "increasing" if engagement_trend > 0 else "decreasing",
                        "percent_change": abs(engagement_trend) * 100
                    },
                    "attendance": {
                        "direction": "increasing" if attendance_trend > 0 else "decreasing",
                        "percent_change": abs(attendance_trend) * 100
                    }
                },
                "student_insights": {
                    "most_improved": most_improved,
                    "most_declined": most_declined
                }
            },
            "session_data": self.timeline_data,
            "engagement_data": {
                "timestamps": timestamps,
                "avg_scores": avg_engagement_scores,
                "attendance": attendance_counts,
                "emotions": emotion_counts
            },
            "student_data": student_engagement_history
        }
        
        return analysis
    
    def _calculate_trend(self, values):
        """Calculate trend direction and magnitude"""
        if not values or len(values) < 2:
            return 0
            
        # Simple trend calculation: (end - start) / start
        start, end = values[0], values[-1]
        if start == 0:  # Avoid division by zero
            return 1 if end > 0 else 0
            
        return (end - start) / start
    
    def generate_visualizations(self):
        """Generate visualizations for timeline data"""
        if not self.timeline_data:
            return None
            
        # Extract data
        timestamps = [entry["timestamp"] for entry in self.timeline_data]
        
        # Format timestamps for better display
        short_timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").strftime("%m-%d %H:%M") 
                           for ts in timestamps]
        
        # Create dataframe for plotting
        data = {
            "Timestamp": short_timestamps,
            "Engagement": [entry["insights"]["engagement"]["average_engagement_score"] 
                          for entry in self.timeline_data],
            "Attendance": [len(entry["insights"]["attendance"]["present_students"]) 
                          for entry in self.timeline_data]
        }
        
        # Add emotions
        emotions = ["happy", "neutral", "sad", "angry", "surprise", "fear", "disgust"]
        for emotion in emotions:
            data[emotion.capitalize()] = []
            
        # Count emotions in each session
        for entry in self.timeline_data:
            emotion_counts = {emotion: 0 for emotion in emotions}
            
            for student in entry["insights"]["student_details"]:
                primary_emotion = student["primary_emotion"]["emotion"]
                if primary_emotion in emotion_counts:
                    emotion_counts[primary_emotion] += 1
            
            # Add to data dictionary
            for emotion in emotions:
                data[emotion.capitalize()].append(emotion_counts[emotion])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create visualizations
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot 1: Engagement over time
        axes[0].plot(df["Timestamp"], df["Engagement"], marker='o', linestyle='-', color='blue')
        axes[0].set_title("Class Engagement Over Time")
        axes[0].set_ylabel("Engagement Score")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True)
        
        # Plot 2: Attendance over time
        axes[1].plot(df["Timestamp"], df["Attendance"], marker='s', linestyle='-', color='green')
        axes[1].set_title("Attendance Over Time")
        axes[1].set_ylabel("Number of Students")
        axes[1].grid(True)
        
        # Plot 3: Emotions over time
        emotion_cols = [col for col in df.columns if col in [e.capitalize() for e in emotions]]
        for emotion in emotion_cols:
            axes[2].plot(df["Timestamp"], df[emotion], marker='.', linestyle='-', label=emotion)
        
        axes[2].set_title("Emotions Over Time")
        axes[2].set_ylabel("Count")
        axes[2].legend()
        axes[2].grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.results_folder, "timeline_analysis.png")
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def export_timeline_data(self):
        """Export timeline data to JSON file"""
        if not self.timeline_data:
            return None
            
        export_path = os.path.join(self.results_folder, "timeline_data.json")
        
        with open(export_path, 'w') as f:
            json.dump(self.timeline_data, f, indent=2)
            
        return export_path