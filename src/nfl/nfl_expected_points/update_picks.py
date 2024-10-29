import papermill as pm
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def main(request_body, client_name):

    logger.info("Executing notebook")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(script_dir, "./notebook.ipynb")
    output_path = "/tmp/output_notebook.ipynb"
    result_path = "/tmp/variable_output.json"

    try:
        pm.execute_notebook(
            notebook_path,
            output_path,
            parameters={
                'current_year': request_body.season,
                'current_week': request_body.week,
                'client_name': client_name
            },
            start_new_kernel=True,
            raise_on_error=False
        )

        logger.info("Notebook executed successfully.")

        # Check if the output file exists and read it
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                output_data = json.load(f)
                logger.info(f"Output data loaded successfully from {result_path}.")

                logger.info("Picks updated")
                logger.info(json.dumps(output_data))
                os.remove(result_path)
                os.remove(output_path)
                # Exit successfully
                return output_data


        else:
            # Log and raise an error if the output file isn't found
            logger.error(f"Output file not found at {result_path}.")
            raise FileNotFoundError(f"Output file not found at {result_path}.")

    except Exception as e:
        # Log any exceptions that occur during notebook execution or file reading
        logger.error(f"Exception during notebook execution: {str(e)}")
        # Re-raise the exception to be caught by the caller
        raise

# if __name__ == "__main__":
#     result = main()

