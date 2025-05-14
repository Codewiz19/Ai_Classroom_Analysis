import os
import pickle
import cv2
import numpy as np
import argparse
import logging
import yaml
from tqdm import tqdm
from pathlib import Path
from deepface import DeepFace
from retinaface import RetinaFace

def setup_logging(log_path):
    """Set up logging"""
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_path, 'embedding_generation.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def detect_face(image, detector_threshold=0.5):
    """Detect face in image using RetinaFace"""
    try:
        faces = RetinaFace.detect_faces(image, threshold=detector_threshold)
        if not faces or isinstance(faces, tuple):
            return None
        
        # Get the first face (assuming one face per image)
        face_data = next(iter(faces.values()))
        facial_area = face_data['facial_area']
        x1, y1, x2, y2 = facial_area
        
        # Extract face ROI
        face_img = image[y1:y2, x1:x2]
        return face_img
    except Exception as e:
        return None

def get_embedding(image, model_name="Facenet512"):
    """Extract embedding from face image"""
    try:
        embedding = DeepFace.represent(
            img_path=image, 
            model_name=model_name,
            enforce_detection=False,
            detector_backend="retinaface"
        )
        return embedding[0]["embedding"]
    except Exception as e:
        return None

def process_student_images(registered_path, detector_threshold=0.5):
    """Process all registered student images and extract embeddings"""
    embeddings = {}
    
    # Get all student folders
    student_dirs = [d for d in os.listdir(registered_path) if os.path.isdir(os.path.join(registered_path, d))]
    
    for student in tqdm(student_dirs, desc="Processing students"):
        student_path = os.path.join(registered_path, student)
        student_embeddings = []
        
        # Get all images for this student
        image_files = [f for f in os.listdir(student_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            try:
                # Load image
                img_path = os.path.join(student_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                
                # Detect face
                face_img = detect_face(image, detector_threshold)
                if face_img is None:
                    continue
                
                # Get embedding
                embedding = get_embedding(face_img)
                if embedding is not None:
                    student_embeddings.append(embedding)
            except Exception as e:
                continue
        
        if student_embeddings:
            # Average the embeddings for this student
            embeddings[student] = np.mean(student_embeddings, axis=0)
    
    return embeddings

def main():
    parser = argparse.ArgumentParser(description="Generate face embeddings for registered students")
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger = setup_logging(config.get('log_path', 'logs'))
    
    logger.info("Starting embedding generation")
    
    # Process student images
    embeddings = process_student_images(
        config.get('registered_path'), 
        config.get('detector_threshold', 0.5)
    )
    
    # Save embeddings
    embeddings_path = config.get('embeddings_path')
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    logger.info(f"Embeddings generated for {len(embeddings)} students and saved to {embeddings_path}")

if __name__ == "__main__":
    main()