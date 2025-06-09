import { Component, signal } from '@angular/core';
import { RouterLink, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { environment } from '../environments/environment';

@Component({
	selector: 'app-root',
	imports: [CommonModule, RouterModule, RouterLink],
	templateUrl: 'app.html',
	styleUrl: 'app.css',
})
export class App {
	token = signal(localStorage.getItem('token'));
	password = signal('');
	loading = signal(false);
	error = signal('');

	constructor(private http: HttpClient) { }
	login() {
		this.loading.set(true);
		this.error.set('');
		const token = this.password();

		this.http
			.post(
				`${environment.apiBase}/auth/login`,
				{ password: token },
				{
					headers: {
						Authorization: token
					}
				}
			)
			.subscribe({
				next: () => {
					localStorage.setItem('token', token);
					this.token.set(token);
					location.reload();
				},
				error: (err) => {
					this.loading.set(false);
					this.error.set(err.status === 401 ? 'Wrong password' : 'Login failed');
				},
			});
	}

}
