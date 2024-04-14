package com.example.diagnosmart.utils

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import retrofit2.HttpException
import retrofit2.Response
import timber.log.Timber
import java.io.IOException

abstract class ResponseHandler{


    suspend fun <T> safeApiCall(apiToBeCalled: suspend () -> Response<T>): AsyncResult<T> {
        return withContext(Dispatchers.IO) {
            try {

                val response: Response<T> = apiToBeCalled()

                if (response.isSuccessful) {
                    Timber.d("Server successful response")
                    AsyncResult.Success(data = response.body()!!)
                } else {
                    Timber.e("Server error = ${response.code()}")

                    when (response.code()) {
                        401 -> {
                        }

                        500 -> {
                        }
                    }

                    AsyncResult.Error(
                            errorMessage = "server error"
                    )
                }

            } catch (e: HttpException) {
                AsyncResult.Error(errorMessage = e.message ?: "Something went wrong")
            } catch (e: IOException) {
                AsyncResult.Error("Please check your network connection")
            } catch (e: Exception) {
                AsyncResult.Error(errorMessage = "Something went wrong")
            }
        }
    }
}