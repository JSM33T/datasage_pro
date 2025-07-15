import { Component, signal } from '@angular/core';
import { NavigationEnd, Router, RouterLink, RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { environment } from '../environments/environment';
import { Collapse } from 'bootstrap';

@Component({
	selector: 'app-root',
	imports: [CommonModule, RouterModule, RouterLink],
	templateUrl: 'app.html',
	styleUrl: 'app.css',
})
export class App {
	token = signal(localStorage.getItem('token'));
	username = signal('');
	password = signal('');
	domain = signal('');
	authType = signal('ldap'); // Default to LDAP
	role = signal(localStorage.getItem('role') || 'user');
	loading = signal(false);
	error = signal('');

	constructor(private http: HttpClient, private router: Router) {
		if (!localStorage.getItem('token')) {
			this.router.navigateByUrl('/');
		}
	}

	ngAfterViewInit(): void {
		this.router.events.subscribe((event) => {
			if (event instanceof NavigationEnd) {
				const nav = document.getElementById('mainNav');
				if (nav && nav.classList.contains('show')) {
					new Collapse(nav).hide();
				}
			}
		});
	}

	confirmLogout() {
		localStorage.removeItem('token');
		localStorage.removeItem('role');
		this.router.navigateByUrl('/');
		location.reload();
	}

	login() {
		this.loading.set(true);
		this.error.set('');

		const loginData = {
			username: this.username(),
			password: this.password(),
			domain: this.domain() || undefined,  // Send undefined if empty
			auth_type: this.authType() // Add authentication type
		};

		this.http
			.post<any>(`${environment.apiBase}/auth/login`, loginData)
			.subscribe({
				next: (response) => {
					localStorage.setItem('token', response.token);
					localStorage.setItem('role', response.role);
					localStorage.setItem('user', JSON.stringify(response.user));

					this.token.set(response.token);
					this.role.set(response.role);

					this.loading.set(false);
					location.reload();
				},
				error: (err) => {
					this.loading.set(false);
					if (err.status === 401) {
						this.error.set('Invalid credentials');
					} else if (err.status === 400) {
						this.error.set('Username and password are required');
					} else {
						this.error.set('Login failed. Please try again.');
					}
				},
			});
	}

	logout() {
		localStorage.removeItem('token');
		localStorage.removeItem('role');
		localStorage.removeItem('user');
		this.token.set(null);
		window.location.href = '/';
	}
}
