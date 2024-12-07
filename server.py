import os
from io import BytesIO
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from fastmcp import FastMCP, Context, Image
from pydantic import BaseModel, Field
import replicate
from PIL import Image as PILImage
import requests

# Load environment variables
load_dotenv()

class ReplicateConfig:
  """Configuration for Replicate API"""
  def __init__(self):
    self.api_token = os.getenv("REPLICATE_API_TOKEN")
    if not self.api_token:
      raise ValueError("REPLICATE_API_TOKEN environment variable is required")
    
    # Initialize client
    self.client = replicate.Client(api_token=self.api_token)
  
  def validate(self):
    """Verify API token works"""
    try:
      self.client.models.list()
      return True
    except Exception as e:
      raise ValueError(f"Failed to validate Replicate API token: {str(e)}")

class GenerationInput(BaseModel):
  """Input model for image generation"""
  model_name: str = Field(..., description="Full model name (e.g. 'stability-ai/sdxl')")
  parameters: Dict[str, Any] = Field(default_factory=dict, description="Model-specific parameters")
  wait_for_completion: bool = Field(default=True, description="Whether to wait for generation to complete")
  output_path: str = Field(default="", description="Path to the output in the model's output schema (e.g. '[0]' for first item in array or empty for direct value)")
  max_size: int = Field(default=800, description="Maximum dimension for the output image")

class SchemaInput(BaseModel):
  """Input model for schema lookup"""
  model_name: str = Field(..., description="Full model name (e.g. 'stability-ai/sdxl')")
  version_id: Optional[str] = Field(None, description="Optional specific version ID")

# Create MCP server
mcp = FastMCP("replicate")

# Initialize config
config = ReplicateConfig()
config.validate()

@mcp.tool()
async def generate_image(input: GenerationInput, ctx: Context) -> Image:
  """Generate an image using a Replicate model
  
  Args:
    input: Generation parameters including model name and parameters
    ctx: Tool context for progress reporting
    
  Returns:
    FastMCP Image object
  """
  try:
    await ctx.report_progress("Starting image generation...")
    
    # Run the model
    output = replicate.run(
      input.model_name,
      input=input.parameters,
      use_file_output=False,  # Get URL string instead of FileOutput
      wait=True  # Wait for completion
    )
    
    if input.wait_for_completion:
      # Get the output image URL based on output_path
      if isinstance(output, list):
        if input.output_path:
          # Parse array index from output_path (e.g. "[0]" -> 0)
          try:
            index = int(input.output_path.strip("[]"))
            image_url = output[index]
          except (ValueError, IndexError):
            raise Exception(f"Invalid output_path: {input.output_path}. For array outputs, use format '[n]' where n is the index.")
        else:
          image_url = output[0]  # Default to first output
      else:
        if input.output_path:
          raise Exception(f"output_path '{input.output_path}' specified but model '{input.model_name}' returns a single output. Remove output_path or leave it empty.")
        image_url = output
        
      await ctx.report_progress("Downloading generated image...")
      
      # Download image
      response = requests.get(image_url, timeout=30)
      response.raise_for_status()
      
      # Process with PIL
      pil_image = PILImage.open(BytesIO(response.content))
      
      # Calculate new size maintaining aspect ratio
      ratio = min(input.max_size / pil_image.width, input.max_size / pil_image.height)
      new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
      
      # Resize with LANCZOS resampling
      pil_image = pil_image.resize(new_size, PILImage.Resampling.LANCZOS)
      
      # Convert to FastMCP Image
      buffered = BytesIO()
      pil_image.save(buffered, format="JPEG", quality=85, optimize=True)
      
      return Image(data=buffered.getvalue(), format="jpeg")
    else:
      return str(output)
      
  except Exception as e:
    raise Exception(f"Image generation failed: {str(e)}")

@mcp.tool()
async def get_model_schema(input: SchemaInput) -> dict:
  """Fetch schema information for a Replicate model
  
  Args:
    input: Schema lookup parameters
    
  Returns:
    Dictionary containing model schema information
  """
  try:
    # Get model
    model = config.client.models.get(input.model_name)
    
    # Get latest version if not specified
    if input.version_id:
      version_id = input.version_id
    else:
      # Get the latest version ID from the model's identifier
      version_id = model.latest_version.id
      
    # Get version details with schema
    version = model.versions.get(version_id)
    
    schema_info = {
      "name": model.name,
      "description": model.description,
      "version": version_id,
      "input_schema": version.openapi_schema["components"]["schemas"]["Input"],
      "output_schema": version.openapi_schema["components"]["schemas"]["Output"]
    }
      
    return schema_info
    
  except Exception as e:
    raise Exception(f"Failed to fetch model schema: {str(e)}") 