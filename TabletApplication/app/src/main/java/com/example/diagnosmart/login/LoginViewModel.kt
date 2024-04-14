package com.example.diagnosmart.login

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.diagnosmart.data.source.repo.AccountRepository
import com.example.diagnosmart.utils.AsyncResult
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject

data class LoginTaskUiState(
        val isTaskCompleted: Boolean = false,
        val isLoading: Boolean = false,
        val userMessage: Int? = null,
        val error: String? = null
)

@HiltViewModel
class LoginViewModel @Inject constructor(
        private val accountRepository: AccountRepository
) : ViewModel() {

    var email by mutableStateOf("")
        private set

    fun updateEmail(input: String) {
        email = input
    }

    var password by mutableStateOf("")
        private set

    fun updatePassword(input: String) {
        password = input
    }

    private val _uiState = MutableStateFlow(LoginTaskUiState())
    val uiState: StateFlow<LoginTaskUiState> = _uiState.asStateFlow()

    init {
        updateEmail("Marta Witherspoon")
        updatePassword("CpwXsQMN")
    }

    fun handleLogin() {
        viewModelScope.launch {

            _uiState.update {
                it.copy(
                        isLoading = true,
                        error = null
                )
            }

            when (val result = accountRepository.login(email, password)) {
                is AsyncResult.Success ->
                    _uiState.update {
                        it.copy(
                                isLoading = false,
                                error = null,
                                isTaskCompleted = true
                        )
                    }

                is AsyncResult.Error -> {
                    _uiState.update {
                        it.copy(
                                isLoading = false,
                                error = result.errorMessage
                        )
                    }
                }

                else -> {}
            }

        }

    }

}