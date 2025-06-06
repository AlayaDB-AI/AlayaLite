import os

# Apache 2.0 License header to be added to the files (Python and C++)
LICENSE_HEADER_PY = """\
# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

LICENSE_HEADER_CPP = """\
/*
 * Copyright 2025 AlayaDB.AI
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
"""

# File extensions to process (Python and C++ files)
EXTENSIONS_PY = {".py"}
EXTENSIONS_CPP = {".c", ".cpp", ".h", ".hpp"}

def add_license_to_file(file_path, header):
    """
    This function adds the license header to the specified file if it doesn't already contain it.
    
    Parameters:
    file_path (str): Path to the file that needs to be processed.
    header (str): License header to be added to the file.
    """
    # Open the file and read its content
    with open(file_path, "r+", encoding="utf-8") as f:
        content = f.read()
        
        # Skip files that already have the license header
        if "Licensed under the Apache License" in content:
            return
        
        # Add the license header at the beginning of the file
        f.seek(0, 0)
        f.write(header + "\n" + content)

def process_directory(directory):
    """
    This function recursively processes the specified directory and adds license headers to files with
    supported extensions (Python and C++ files).
    
    Parameters:
    directory (str): The directory to start processing from.
    """
    # Walk through the directory and process files
    for root, _, files in os.walk(directory):
        for file in files:
            # If the file is a Python file, add Python license header
            if any(file.endswith(ext) for ext in EXTENSIONS_PY):
                add_license_to_file(os.path.join(root, file), LICENSE_HEADER_PY)
            
            # If the file is a C++ file, add C++ license header
            elif any(file.endswith(ext) for ext in EXTENSIONS_CPP):
                add_license_to_file(os.path.join(root, file), LICENSE_HEADER_CPP)

if __name__ == "__main__":
    # Start processing from the current directory
    process_directory("../python")
    process_directory("../include")
    process_directory("../tests")
