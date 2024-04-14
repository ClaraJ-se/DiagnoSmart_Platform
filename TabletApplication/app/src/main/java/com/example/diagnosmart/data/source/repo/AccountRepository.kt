package com.example.diagnosmart.data.source.repo

import com.example.diagnosmart.data.source.models.User
import com.example.diagnosmart.utils.AsyncResult


interface AccountRepository {

    suspend fun isSessionActive(): Boolean

    suspend fun login(email: String, password: String): AsyncResult<User>

    suspend fun exit(): AsyncResult<Any>

}