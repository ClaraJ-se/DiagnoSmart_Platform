package com.example.diagnosmart.utils

sealed class AsyncResult<T> {

    class Success<T>(val data: T) : AsyncResult<T>()

    class Error<T>(val errorMessage: String) : AsyncResult<T>()

    class Loading<T> : AsyncResult<T>()
}