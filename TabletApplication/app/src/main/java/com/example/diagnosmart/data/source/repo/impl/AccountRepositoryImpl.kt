package com.example.diagnosmart.data.source.repo.impl

import android.util.Log
import com.example.diagnosmart.data.source.local.auth.AuthManager
import com.example.diagnosmart.data.source.local.user.UserDao
import com.example.diagnosmart.data.source.models.User
import com.example.diagnosmart.data.source.models.toLocal
import com.example.diagnosmart.data.source.repo.AccountRepository
import com.example.diagnosmart.utils.AsyncResult
import com.example.diagnosmart.utils.ResponseHandler
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import timber.log.Timber
import javax.inject.Inject

class AccountRepositoryImpl @Inject constructor(
        private val userDao: UserDao,
        private val authManager: AuthManager
) : ResponseHandler(), AccountRepository {


    override suspend fun isSessionActive(): Boolean {
        return withContext(Dispatchers.IO) {

            !authManager.email.isNullOrEmpty()
        }
    }

    override suspend fun login(email: String, password: String): AsyncResult<User> {
        return withContext(Dispatchers.IO) {

            val user = User(firstName = email)

            authManager.email = email
            authManager.password = password

            userDao.deleteAll()
            userDao.insert(user.toLocal())

            return@withContext AsyncResult.Success(user)
        }
    }

    override suspend fun exit(): AsyncResult<Any> {
        return withContext(Dispatchers.IO) {
            try {
                authManager.clearTokenData()

                AsyncResult.Success(Any())
            } catch (exception: Exception) {
                Timber.e("AccountRepositoryImpl = ${exception.localizedMessage}")
                AsyncResult.Error(errorMessage = "Something went wrong")
            }
        }
    }

}
