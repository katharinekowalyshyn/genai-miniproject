"""
Grading Bot - A prototype grading system using LLM Proxy with RAG.

This system allows TAs/Professors to:
1. Upload course materials (syllabus, assignments, solutions, lectures, textbook)
2. Grade student submissions using RAG to retrieve relevant context
3. Get detailed feedback and scores based on course materials

Users are distinguished by session_id to maintain separate document collections.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from time import sleep
import json
from llmproxy import LLMProxy


class GradingBot:
    """
    A grading bot that uses LLM Proxy with RAG to grade student submissions
    based on course materials.
    """
    
    def __init__(
        self,
        session_id: str,
        model: str = "4o-mini",
    ):
        """
        Initialize the GradingBot.
        
        Args:
            session_id: Unique identifier for this TA/Professor's session.
                       Documents uploaded will be associated with this session.
            model: LLM model to use for grading (default: "4o-mini")
        """
        self.client = LLMProxy()
        self.session_id = session_id
        self.model = model
        # Fixed RAG and temperature parameters
        self.rag_threshold = 0.3
        self.rag_k = 5
        self.temperature = 0.0
        
        # Track uploaded documents
        self.uploaded_docs: List[Dict[str, str]] = []
    
    def upload_syllabus(self, file_path: Union[str, Path], description: Optional[str] = None) -> Dict:
        """
        Upload the course syllabus.
        
        Args:
            file_path: Path to the syllabus PDF file
            description: Optional description of the document
            
        Returns:
            Response from upload operation
        """
        result = self.client.upload_file(
            file_path=file_path,
            session_id=self.session_id,
            description=description or "Course Syllabus",
            strategy="smart"
        )
        if "error" not in result:
            self.uploaded_docs.append({
                "type": "syllabus",
                "path": str(file_path),
                "description": description or "Course Syllabus"
            })
        return result
    
    def upload_homework_assignment(
        self,
        file_path: Union[str, Path],
        assignment_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict:
        """
        Upload a homework assignment.
        
        Args:
            file_path: Path to the assignment PDF file
            assignment_name: Name of the assignment (e.g., "HW1", "Homework 2")
            description: Optional description
            
        Returns:
            Response from upload operation
        """
        desc = description or f"Homework Assignment: {assignment_name or Path(file_path).stem}"
        result = self.client.upload_file(
            file_path=file_path,
            session_id=self.session_id,
            description=desc,
            strategy="smart"
        )
        if "error" not in result:
            self.uploaded_docs.append({
                "type": "homework_assignment",
                "path": str(file_path),
                "assignment_name": assignment_name,
                "description": desc
            })
        return result
    
    def upload_homework_solution(
        self,
        file_path: Union[str, Path],
        assignment_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict:
        """
        Upload a homework solution/answer key.
        
        Args:
            file_path: Path to the solution PDF file
            assignment_name: Name of the assignment this solution corresponds to
            description: Optional description
            
        Returns:
            Response from upload operation
        """
        desc = description or f"Homework Solution: {assignment_name or Path(file_path).stem}"
        result = self.client.upload_file(
            file_path=file_path,
            session_id=self.session_id,
            description=desc,
            strategy="smart"
        )
        if "error" not in result:
            self.uploaded_docs.append({
                "type": "homework_solution",
                "path": str(file_path),
                "assignment_name": assignment_name,
                "description": desc
            })
        return result
    
    def upload_lecture_material(
        self,
        file_path: Union[str, Path],
        lecture_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict:
        """
        Upload lecture slides or reading materials.
        
        Args:
            file_path: Path to the lecture PDF file
            lecture_name: Name/title of the lecture
            description: Optional description
            
        Returns:
            Response from upload operation
        """
        desc = description or f"Lecture Material: {lecture_name or Path(file_path).stem}"
        result = self.client.upload_file(
            file_path=file_path,
            session_id=self.session_id,
            description=desc,
            strategy="smart"
        )
        if "error" not in result:
            self.uploaded_docs.append({
                "type": "lecture_material",
                "path": str(file_path),
                "lecture_name": lecture_name,
                "description": desc
            })
        return result
    
    def upload_textbook(self, file_path: Union[str, Path], description: Optional[str] = None) -> Dict:
        """
        Upload the course textbook.
        
        Args:
            file_path: Path to the textbook PDF file
            description: Optional description
            
        Returns:
            Response from upload operation
        """
        result = self.client.upload_file(
            file_path=file_path,
            session_id=self.session_id,
            description=description or "Course Textbook",
            strategy="smart"
        )
        if "error" not in result:
            self.uploaded_docs.append({
                "type": "textbook",
                "path": str(file_path),
                "description": description or "Course Textbook"
            })
        return result
    
    def wait_for_processing(self, seconds: int = 20):
        """
        Wait for uploaded documents to be processed by the backend.
        
        Args:
            seconds: Number of seconds to wait (default: 20)
        """
        sleep(seconds)
    
    def _format_rag_context(self, rag_context: List[Dict]) -> str:
        """
        Format RAG context into a readable string for the LLM.
        
        Args:
            rag_context: List of retrieved context chunks from RAG
            
        Returns:
            Formatted context string
        """
        if not rag_context:
            return ""
        
        context_string = "The following context from course materials may be helpful:\n\n"
        
        for i, collection in enumerate(rag_context, 1):
            doc_summary = collection.get('doc_summary', '')
            chunks = collection.get('chunks', [])
            
            if doc_summary:
                context_string += f"[Document {i}]: {doc_summary}\n"
            
            for j, chunk in enumerate(chunks, 1):
                context_string += f"  {i}.{j}. {chunk}\n"
            
            context_string += "\n"
        
        return context_string
    
    def grade_submission(
        self,
        question: str,
        student_answer: str,
        max_points: Optional[float] = None,
        rubric: Optional[str] = None,
        assignment_name: Optional[str] = None,
        wait_after_upload: bool = True
    ) -> Dict:
        """
        Grade a student submission using RAG to retrieve relevant course materials.
        
        Args:
            question: The question or problem statement
            student_answer: The student's answer to grade
            max_points: Maximum points for this question (optional)
            rubric: Additional grading rubric or instructions (optional)
            assignment_name: Name of the assignment (for context)
            wait_after_upload: Whether to wait after uploading (if student_answer is a file)
            
        Returns:
            Dictionary containing:
                - score: Numerical score (if max_points provided)
                - feedback: Detailed feedback
                - rag_context_used: Context retrieved from course materials
                - raw_response: Full LLM response
        """
        # Build the grading query with context
        query_parts = []
        
        if assignment_name:
            query_parts.append(f"Assignment: {assignment_name}")
        
        query_parts.append(f"Question: {question}")
        query_parts.append(f"\nStudent Answer:\n{student_answer}")
        
        if rubric:
            query_parts.append(f"\nGrading Rubric:\n{rubric}")
        
        if max_points:
            query_parts.append(f"\nMaximum Points: {max_points}")
        
        query = "\n".join(query_parts)
        
        # Retrieve relevant context from course materials
        rag_result = self.client.retrieve(
            query=query,
            session_id=self.session_id,
            rag_threshold=self.rag_threshold,
            rag_k=self.rag_k
        )
        
        # Check for errors in retrieval
        if "error" in rag_result:
            return {
                "error": f"RAG retrieval failed: {rag_result['error']}",
                "raw_response": rag_result
            }
        
        # Extract RAG context
        rag_context = rag_result.get("rag_context", [])
        formatted_context = self._format_rag_context(rag_context)
        
        # Build system prompt for grading
        system_prompt = """You are an expert teaching assistant grading a student submission for a Discrete Math course.

Your task is to:
1. Evaluate the student's answer for correctness, completeness, and clarity
2. Compare it against the course materials and solutions provided in the context
3. Provide constructive feedback highlighting what the student did well and what needs improvement
4. Assign a score if maximum points are specified

Guidelines:
- Be fair and consistent in your grading
- Reference specific course materials when relevant
- Provide specific, actionable feedback
- If the answer is partially correct, explain what parts are correct and what needs work
- Consider mathematical rigor, notation, and explanation quality
- If the answer is incorrect, guide the student toward the correct approach without giving away the full solution

Format your response as:
SCORE: [X/Y points] (if max_points provided)
FEEDBACK:
[Detailed feedback here]"""
        
        # Combine query with RAG context
        if formatted_context:
            full_query = f"{formatted_context}\n\n{query}"
        else:
            full_query = query
        
        # Generate grading using LLM with RAG
        response = self.client.generate(
            model=self.model,
            system=system_prompt,
            query=full_query,
            temperature=self.temperature,
            session_id=self.session_id,
            rag_usage=False,  # We're manually including context
            rag_threshold=self.rag_threshold,
            rag_k=self.rag_k
        )
        
        if "error" in response:
            return {
                "error": f"Grading generation failed: {response['error']}",
                "raw_response": response
            }
        
        # Extract result text
        result_text = response.get("result", "")
        
        # Try to parse score if max_points provided
        score = None
        if max_points and result_text:
            # Look for score pattern like "SCORE: X/Y" or "X/Y points"
            import re
            score_match = re.search(r'(\d+\.?\d*)\s*/\s*(\d+\.?\d*)', result_text)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    max_pts = float(score_match.group(2))
                    # Normalize if needed
                    if max_pts != max_points:
                        score = (score / max_pts) * max_points
                except ValueError:
                    pass
        
        return {
            "score": score,
            "max_points": max_points,
            "feedback": result_text,
            "rag_context_used": formatted_context if formatted_context else "No relevant context retrieved",
            "raw_response": response
        }
    
    def grade_from_file(
        self,
        question: str,
        student_answer_file: Union[str, Path],
        max_points: Optional[float] = None,
        rubric: Optional[str] = None,
        assignment_name: Optional[str] = None
    ) -> Dict:
        """
        Grade a student submission from a file.
        
        Args:
            question: The question or problem statement
            student_answer_file: Path to file containing student's answer
            max_points: Maximum points for this question (optional)
            rubric: Additional grading rubric or instructions (optional)
            assignment_name: Name of the assignment (for context)
            
        Returns:
            Same as grade_submission()
        """
        path = Path(student_answer_file)
        if not path.exists():
            return {"error": f"File not found: {path}"}
        
        # Read the file content
        try:
            with open(path, 'r', encoding='utf-8') as f:
                student_answer = f.read()
        except Exception as e:
            return {"error": f"Error reading file: {e}"}
        
        return self.grade_submission(
            question=question,
            student_answer=student_answer,
            max_points=max_points,
            rubric=rubric,
            assignment_name=assignment_name
        )
    
    def get_uploaded_documents(self) -> List[Dict[str, str]]:
        """
        Get list of documents that have been uploaded to this session.
        
        Returns:
            List of document metadata dictionaries
        """
        return self.uploaded_docs.copy()


# Example usage and CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grading Bot - Grade student submissions using LLM with RAG")
    parser.add_argument("--session-id", type=str, required=True,
                       help="Session ID for this TA/Professor (e.g., 'ta_john_doe')")
    parser.add_argument("--upload", type=str, choices=["syllabus", "assignment", "solution", "lecture", "textbook"],
                       help="Upload a document type")
    parser.add_argument("--file", type=str, help="Path to PDF file to upload")
    parser.add_argument("--description", type=str, help="Description for uploaded document")
    parser.add_argument("--grade", action="store_true", help="Grade a submission")
    parser.add_argument("--question", type=str, help="Question/problem statement")
    parser.add_argument("--answer", type=str, help="Student's answer (text or file path)")
    parser.add_argument("--max-points", type=float, help="Maximum points for the question")
    parser.add_argument("--rubric", type=str, help="Grading rubric (text or file path)")
    parser.add_argument("--assignment", type=str, help="Assignment name")
    parser.add_argument("--model", type=str, default="4o-mini", help="LLM model to use")
    parser.add_argument("--wait", type=int, default=20, help="Seconds to wait after upload")
    
    args = parser.parse_args()
    
    bot = GradingBot(session_id=args.session_id, model=args.model)
    
    if args.upload:
        if not args.file:
            print("Error: --file required when using --upload")
            exit(1)
        
        print(f"Uploading {args.upload} from {args.file}...")
        
        if args.upload == "syllabus":
            result = bot.upload_syllabus(args.file, args.description)
        elif args.upload == "assignment":
            result = bot.upload_homework_assignment(args.file, args.assignment, args.description)
        elif args.upload == "solution":
            result = bot.upload_homework_solution(args.file, args.assignment, args.description)
        elif args.upload == "lecture":
            result = bot.upload_lecture_material(args.file, args.assignment, args.description)
        elif args.upload == "textbook":
            result = bot.upload_textbook(args.file, args.description)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            exit(1)
        else:
            print(f"Success! Waiting {args.wait} seconds for processing...")
            bot.wait_for_processing(args.wait)
            print("Upload complete!")
    
    elif args.grade:
        if not args.question or not args.answer:
            print("Error: --question and --answer required when using --grade")
            exit(1)
        
        # Check if answer is a file path
        answer_path = Path(args.answer)
        if answer_path.exists():
            student_answer = answer_path.read_text(encoding='utf-8')
        else:
            student_answer = args.answer
        
        # Check if rubric is a file path
        rubric_text = None
        if args.rubric:
            rubric_path = Path(args.rubric)
            if rubric_path.exists():
                rubric_text = rubric_path.read_text(encoding='utf-8')
            else:
                rubric_text = args.rubric
        
        print("Grading submission...")
        result = bot.grade_submission(
            question=args.question,
            student_answer=student_answer,
            max_points=args.max_points,
            rubric=rubric_text,
            assignment_name=args.assignment
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            exit(1)
        
        print("\n" + "="*60)
        print("GRADING RESULT")
        print("="*60)
        if result.get("score") is not None:
            print(f"\nSCORE: {result['score']:.2f} / {result['max_points']:.2f} points")
        print(f"\nFEEDBACK:\n{result['feedback']}")
        print("\n" + "="*60)
        print(f"\nRAG Context Used:\n{result['rag_context_used']}")
    
    else:
        print("Uploaded documents:")
        for doc in bot.get_uploaded_documents():
            print(f"  - {doc['type']}: {doc.get('description', 'N/A')}")

