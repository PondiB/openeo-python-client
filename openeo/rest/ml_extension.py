from __future__ import annotations

import logging
import pathlib
import typing
from typing import Optional, Union, List

from openeo.internal.documentation import openeo_process
from openeo.internal.graph_building import PGNode
from openeo.rest import (
    DEFAULT_JOB_STATUS_POLL_CONNECTION_RETRY_INTERVAL,
    DEFAULT_JOB_STATUS_POLL_INTERVAL_MAX,
)
from openeo.rest._datacube import _ProcessGraphAbstraction
from openeo.rest.job import BatchJob
from openeo.util import dict_no_none
from openeo.rest.datacube import DataCube

if typing.TYPE_CHECKING:
    # Imports for type checking only (circular import issue at runtime).
    from openeo import Connection
    

_log = logging.getLogger(__name__)


class MLModel(_ProcessGraphAbstraction):
    """
    A machine learning model.

    It is the result of a training procedure, e.g. output of a ``fit_...`` process,
    and can be used for prediction (classification or regression) with the corresponding ``predict_...`` process.

    .. versionadded:: 0.10.0
    """

    def __init__(self, graph: PGNode, connection: Union[Connection, None]):
        super().__init__(pgnode=graph, connection=connection)

    def save_ml_model(self, name: str, options: Optional[dict] = None) -> MLModel:
        """
        Save a ML model.
        Saves a machine learning model as part of a batch job.
        The model will be accompanied by a separate STAC Item that implements the mlm-model extension.

        :param name: A distinct name of the model.
        :param options: Additional parameters to create the file(s).
        :return: Returns false if the process failed to store the model, true otherwise.

        .. warning:: EXPERIMENTAL: this process is experimental with the potential for major things to change.
        """
        pgnode = PGNode(
            process_id="save_ml_model",
            arguments={
                "data": self,
                "name": name,
                "options": options or {}
            }
        )
        return MLModel(graph=pgnode, connection=self._connection)

    @staticmethod
    @openeo_process
    def load_ml_model(connection: Connection, id: Union[str, BatchJob]) -> MLModel:
        """
        Loads a machine learning model from a STAC Item.

        :param connection: connection object
        :param id: STAC item reference, as URL, batch job (id) or user-uploaded file
        :return:

        .. versionadded:: 0.10.0
        """
        if isinstance(id, BatchJob):
            id = id.job_id
        return MLModel(graph=PGNode(process_id="load_ml_model", id=id), connection=connection)
        
    @staticmethod
    @openeo_process
    def load_stac_ml(
        connection: Connection,
        uri: Union[str, pathlib.Path],
        model_asset: Optional[str] = None,
        input_index: int = 0,
        output_index: int = 0
    ) -> MLModel:
        """
        Load a machine learning model from a STAC Item.

        Loads a machine learning model from a STAC Item. Such a model could be trained and saved as part of a previous batch
        job with processes such as ``ml_fit()`` and ``save_ml_model()`` or externally hosted models.

        :param connection: Connection object
        :param uri: The STAC Item to load the machine learning model from. The STAC Item must implement the
            `mlm <https://github.com/stac-extensions/mlm>`_ extension. This parameter can point to a remote STAC Item
            via URL or a local JSON file.
        :param model_asset: The Asset name of the given STAC Item which represents the actual ML model. The asset must list
            ``mlm:model`` as its role. If only one asset lists ``mlm:model`` as its role, this parameter is optional as
            this asset will be used by default. If multiple assets list ``mlm:model`` as their role, this parameter is
            required to determine which asset to use.
        :param input_index: STAC:MLM items support multiple ML model input specifications. This parameter specifies the
            index of the input specification in the ``mlm:input`` array to use for prediction or training. As
            ``mlm:input`` is an array, the first input in the array has index 0.
        :param output_index: STAC:MLM items support multiple ML model output specifications. This parameter specifies the
            index of the output specification in the ``mlm:output`` array to use for prediction or training. As
            ``mlm:output`` is an array, the first output in the array has index 0.
        :return: A machine learning model to be used with machine learning processes such as ``ml_predict()``.

        .. warning:: EXPERIMENTAL: this process is experimental with the potential for major things to change.
        """
        arguments = {
            "uri": str(uri),
            "input_index": input_index,
            "output_index": output_index
        }
        
        if model_asset is not None:
            arguments["model_asset"] = model_asset
            
        return MLModel(graph=PGNode(process_id="load_stac_ml", arguments=arguments), connection=connection)

    def execute_batch(
        self,
        outputfile: Union[str, pathlib.Path],
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        plan: Optional[str] = None,
        budget: Optional[float] = None,
        print=print,
        max_poll_interval: float = DEFAULT_JOB_STATUS_POLL_INTERVAL_MAX,
        connection_retry_interval: float = DEFAULT_JOB_STATUS_POLL_CONNECTION_RETRY_INTERVAL,
        additional: Optional[dict] = None,
        job_options: Optional[dict] = None,
        show_error_logs: bool = True,
        log_level: Optional[str] = None,
    ) -> BatchJob:
        """
        Execute the underlying process graph at the backend in batch job mode:

        - create the job (like :py:meth:`create_job`)
        - start the job (like :py:meth:`BatchJob.start() <openeo.rest.job.BatchJob.start>`)
        - track the job's progress with an active polling loop
          (like :py:meth:`BatchJob.run_synchronous() <openeo.rest.job.BatchJob.run_synchronous>`)
        - optionally (if ``outputfile`` is specified) download the job's results
          when the job finished successfully

        .. note::
            Because of the active polling loop,
            which blocks any further progress of your script or application,
            this :py:meth:`execute_batch` method is mainly recommended
            for batch jobs that are expected to complete
            in a time that is reasonable for your use case.

        :param outputfile: (optional) output path to download to.
        :param title: (optional) job title.
        :param description: (optional) job description.
        :param plan: (optional) the billing plan to process and charge the job with.
        :param budget: (optional) maximum budget to be spent on executing the job.
            Note that some backends do not honor this limit.
        :param additional: (optional) additional (top-level) properties to set in the request body
        :param job_options: (optional) dictionary of job options to pass to the backend
            (under top-level property "job_options")
        :param show_error_logs: whether to automatically print error logs when the batch job failed.
        :param log_level: (optional) minimum severity level for log entries that the back-end should keep track of.
            One of "error" (highest severity), "warning", "info", and "debug" (lowest severity).
        :param max_poll_interval: maximum number of seconds to sleep between job status polls
        :param connection_retry_interval: how long to wait when status poll failed due to connection issue
        :param print: print/logging function to show progress/status

        :return: Handle to the job created at the backend.

        .. versionchanged:: 0.36.0
            Added argument ``additional``.

        .. versionchanged:: 0.37.0
            Added argument ``show_error_logs``.

        .. versionchanged:: 0.37.0
            Added argument ``log_level``.
        """
        job = self.create_job(
            title=title,
            description=description,
            plan=plan,
            budget=budget,
            additional=additional,
            job_options=job_options,
            log_level=log_level,
        )
        job.start_and_wait(
            print=print,
            max_poll_interval=max_poll_interval,
            connection_retry_interval=connection_retry_interval,
            show_error_logs=show_error_logs,
        )
        if outputfile is not None:
            job.download_result(target=outputfile)
        return job

    def create_job(
        self,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        plan: Optional[str] = None,
        budget: Optional[float] = None,
        additional: Optional[dict] = None,
        job_options: Optional[dict] = None,
        log_level: Optional[str] = None,
    ) -> BatchJob:
        """
        Send the underlying process graph to the backend
        to create an openEO batch job
        and return a corresponding :py:class:`~openeo.rest.job.BatchJob` instance.

        Note that this method only *creates* the openEO batch job at the backend,
        but it does not *start* it.
        Use :py:meth:`execute_batch` instead to let the openEO Python client
        take care of the full job life cycle: create, start and track its progress until completion.


        :param title: (optional) job title.
        :param description: (optional) job description.
        :param plan: (optional) the billing plan to process and charge the job with.
        :param budget: (optional) maximum budget to be spent on executing the job.
            Note that some backends do not honor this limit.
        :param additional: (optional) additional (top-level) properties to set in the request body
        :param job_options: (optional) dictionary of job options to pass to the backend
            (under top-level property "job_options")
        :param log_level: (optional) minimum severity level for log entries that the back-end should keep track of.
            One of "error" (highest severity), "warning", "info", and "debug" (lowest severity).

        :return: Handle to the job created at the backend.

        .. versionchanged:: 0.36.0
            Added argument ``additional``.

        .. versionchanged:: 0.37.0
            Added argument ``log_level``.
        """
        # TODO: centralize `create_job` for `DataCube`, `VectorCube`, `MlModel`, ...
        pg = self
        if pg.result_node().process_id not in {"save_ml_model"}:
            _log.warning("Process graph has no final `save_ml_model`. Adding it automatically.")
            pg = pg.save_ml_model()
        return self._connection.create_job(
            process_graph=pg.flat_graph(),
            title=title,
            description=description,
            plan=plan,
            budget=budget,
            additional=additional,
            job_options=job_options,
            log_level=log_level,
        )
    

    def fit(self, training_set, target: str) -> MLModel: # ml_fit
        """
        Train a machine learning model.
        Executes the fit of a specified machine learning model based on training data.
        The function is generic and supports different machine learning models.

        :param training_set: The training set for the model, provided as a vector data cube. 
            This set contains both the independent variables and the dependent variable that 
            the model analyzes to learn patterns and relationships within the data.
        :param target: The column in the training set that represents the dependent variable for model training.
        :return: A trained model object that can be saved with save_ml_model() and restored with load_ml_model().

        .. warning:: EXPERIMENTAL: this process is experimental with the potential for major things to change.
        """
        pgnode = PGNode(
            process_id="ml_fit",
            arguments=dict_no_none(
                model=self,
                training_set=training_set,
                target=target
            ),
        )
        model = MLModel(graph=pgnode, connection=self._connection)
        return model

    def predict(self, data, dimensions: Optional[List] = None) -> DataCube: # ml_predict
        """
        Predict using ML.
        Applies a machine learning model to a data cube of input features and returns the predicted values.

        :param data: The data cube containing the input features.
        :param dimensions: Zero or more dimensions that will be reduced by the model. 
            Fails with a DimensionNotAvailable exception if one of the specified dimensions does not exist.
        :return: A data cube with the predicted values. It removes the specified dimensions and adds new dimension 
            for the predicted values. It has the name predictions and is of type other. If a single value is returned, 
            the dimension has a single label with name 0.

        .. warning:: EXPERIMENTAL: this process is experimental with the potential for major things to change.
        """
        pgnode = PGNode(
            process_id="ml_predict",
            arguments=dict_no_none(
                model=self,
                data=data,
                dimensions=dimensions
            ),
        )
        datacube = DataCube(graph=pgnode, connection=self._connection)
        return datacube
    
    def predict_probabilities(self, data, dimensions: Optional[List] = None) -> DataCube: # ml_predict_probabilities
        """
        Predict class probabilities using ML.
        Applies a machine learning model to a data cube of input features and returns the predicted class probabilities.

        :param data: The data cube containing the input features.
        :param dimensions: Zero or more dimensions that will be reduced by the model. 
            Fails with a DimensionNotAvailable exception if one of the specified dimensions does not exist.
        :return: A data cube with the predicted class probabilities. It removes the specified dimensions and adds 
            a new dimension for the class probabilities. The dimension has the name classes and is of type other. 
            Each label in the dimension represents a class, and the values are the probabilities for each class.

        .. warning:: EXPERIMENTAL: this process is experimental with the potential for major things to change.
        """
        pgnode = PGNode(
            process_id="ml_predict_probabilities",
            arguments=dict_no_none(
                model=self,
                data=data,
                dimensions=dimensions
            ),
        )
        datacube = DataCube(graph=pgnode, connection=self._connection)
        return datacube
